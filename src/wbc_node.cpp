#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "legged_wbc/pinocchio_interface.hpp"
#include "legged_wbc/wbc_solver.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <memory>
#include <map>
#include <Eigen/Geometry>

namespace legged_wbc {

class WbcNode : public rclcpp::Node {
public:
    WbcNode() : Node("wbc_node"), state_received_(false), imu_received_(false), pose_received_(false) {
        this->declare_parameter("urdf_path", "/tmp/hunter_generated.urdf");
        this->declare_parameter("control_frequency", 500.0);
        
        std::string urdf_path = this->get_parameter("urdf_path").as_string();
        double freq = this->get_parameter("control_frequency").as_double();
        
        pinocchio_ = std::make_shared<PinocchioInterface>();
        if (!pinocchio_->loadURDF(urdf_path)) {
            throw std::runtime_error("Failed to load URDF");
        }
        
        solver_ = std::make_unique<WbcSolver>(pinocchio_);
        
        int nq = pinocchio_->getModel().nq;
        int nv = pinocchio_->getModel().nv;
        current_q_ = Eigen::VectorXd::Zero(nq);
        current_v_ = Eigen::VectorXd::Zero(nv);
        current_q_(6) = 1.0; 
        
        target_q_ = Eigen::VectorXd::Zero(nq);
        target_q_(6) = 1.0;
        target_q_set_ = false;
        
        contact_status_.resize(4, false);
        contact_forces_.resize(4, 0.0);

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&WbcNode::jointStateCallback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_imu_sensor/imu", 10, std::bind(&WbcNode::imuCallback, this, std::placeholders::_1));
            
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/base_link_pose_sensor/pose", 10, std::bind(&WbcNode::poseCallback, this, std::placeholders::_1));
            
        setupWrenchSubscriptions();
        
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/forward_command_controller/commands", 10);
        
        auto period = std::chrono::duration<double>(1.0 / freq);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&WbcNode::controlLoop, this));
            
        RCLCPP_INFO(this->get_logger(), "WBC Node Started (Contact-Aware)");
    }

private:
    void setupWrenchSubscriptions() {
        std::string frames[] = {"leg_l_f1_site", "leg_l_f2_site", "leg_r_f1_site", "leg_r_f2_site"};
        for (int i = 0; i < 4; ++i) {
            std::string topic = "/" + frames[i] + "_wrench_sensor/wrench";
            wrench_subs_[i] = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                topic, 10, [this, i](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
                    this->contact_forces_[i] = std::abs(msg->wrench.force.z); // use abs for different orientations
                    this->contact_status_[i] = (this->contact_forces_[i] > 3.0); // 3N threshold
                });
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        static rclcpp::Time last_time = this->get_clock()->now();
        static Eigen::Vector3d last_pos(0,0,0);
        
        rclcpp::Time current_time = this->get_clock()->now();
        double dt = (current_time - last_time).seconds();
        
        current_q_(0) = msg->pose.position.x;
        current_q_(1) = msg->pose.position.y;
        current_q_(2) = msg->pose.position.z;
        current_q_(3) = msg->pose.orientation.x;
        current_q_(4) = msg->pose.orientation.y;
        current_q_(5) = msg->pose.orientation.z;
        current_q_(6) = msg->pose.orientation.w;

        if (dt > 0.0001 && dt < 1.0) {
            Eigen::Vector3d curr_pos(current_q_(0), current_q_(1), current_q_(2));
            if (last_pos.norm() > 0) {
                Eigen::Vector3d vel_world = (curr_pos - last_pos) / dt;
                
                // Pinocchio FreeFlyer expects velocity in LOCAL frame
                Eigen::Quaterniond q(current_q_(6), current_q_(3), current_q_(4), current_q_(5));
                Eigen::Vector3d vel_local = q.inverse() * vel_world;

                // Basic low-pass filter
                current_v_(0) = 0.7 * current_v_(0) + 0.3 * vel_local.x();
                current_v_(1) = 0.7 * current_v_(1) + 0.3 * vel_local.y();
                current_v_(2) = 0.7 * current_v_(2) + 0.3 * vel_local.z();
            }
            last_pos = curr_pos;
        }
        last_time = current_time;
        pose_received_ = true;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        current_v_(3) = msg->angular_velocity.x;
        current_v_(4) = msg->angular_velocity.y;
        current_v_(5) = msg->angular_velocity.z;
        imu_received_ = true;
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (joint_name_to_index_.empty()) {
            for (size_t i = 0; i < msg->name.size(); ++i) joint_name_to_index_[msg->name[i]] = i;
        }

        std::string robot_joints[] = {
            "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
            "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"
        };

        for (int i = 0; i < 10; ++i) {
            auto it = joint_name_to_index_.find(robot_joints[i]);
            if (it != joint_name_to_index_.end()) {
                int pin_id = pinocchio_->getJointId(robot_joints[i]);
                if (pin_id >= 0) {
                    current_q_(pinocchio_->getJointIdxQ(pin_id)) = msg->position[it->second];
                    current_v_(pinocchio_->getJointIdxV(pin_id)) = msg->velocity[it->second];
                }
            }
        }
        
        if (!target_q_set_ && pose_received_) {
            target_q_ = current_q_;
            target_q_(3) = 0.0; target_q_(4) = 0.0; target_q_(5) = 0.0; target_q_(6) = 1.0;
            // Target crouch posture
            target_q_(10) = 0.8; target_q_(15) = 0.8; // knees
            target_q_(9) = -0.4; target_q_(14) = -0.4; // hips
            target_q_set_ = true;
            RCLCPP_INFO(this->get_logger(), "WBC: Target Initialized at height %.3f", current_q_(2));
        }
        state_received_ = true;
    }

    void controlLoop() {
        if (!state_received_ || !imu_received_ || !pose_received_) return;

        // Dynamic target Z for squatting test
        static bool stabilized = false;
        double t = this->get_clock()->now().seconds();
        
        int contact_count = 0;
        for (bool b : contact_status_) if (b) contact_count++;
        if (contact_count >= 3) stabilized = true;

        if (target_q_set_) {
            if (!stabilized) {
                // Keep spawning height until stable
            } else {
                // Smoothly lower to 0.55m then squat
                static double start_time = t;
                double elapsed = t - start_time;
                double target_h = 0.55;
                if (elapsed < 5.0) {
                    double alpha = elapsed / 5.0;
                    target_q_(2) = (1.0 - alpha) * current_q_(2) + alpha * target_h;
                } else {
                    target_q_(2) = target_h; // Hold fixed height
                }
            }
        }

        pinocchio_->update(current_q_, current_v_);

        // Contact heuristic: if force is low but foot is very close to ground, assume contact during init/falling
        auto& model = pinocchio_->getModel();
        auto& data = pinocchio_->getData();
        for (int i = 0; i < 4; ++i) {
            std::string frame_names[] = {"leg_l_f1_link", "leg_l_f2_link", "leg_r_f1_link", "leg_r_f2_link"};
            auto frame_id = model.getFrameId(frame_names[i]);
            double foot_z = data.oMf[frame_id].translation().z();
            if (foot_z < 0.02) { // 2cm threshold
                contact_status_[i] = true;
            }
        }
        auto result = solver_->solve(current_q_, current_v_, target_q_, contact_status_);
        if (result.success) {
            auto cmd = std_msgs::msg::Float64MultiArray();
            cmd.data.resize(10);
            for (int i = 0; i < 10; ++i) cmd.data[i] = std::clamp(result.tau(i), -60.0, 60.0);
            effort_pub_->publish(cmd);
            
            static int count = 0;
            if (count++ % 100 == 0) {
                Eigen::Quaterniond q(current_q_(6), current_q_(3), current_q_(4), current_q_(5));
                auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
                int c_count = 0;
                for(bool b : contact_status_) if(b) c_count++;
                RCLCPP_INFO(this->get_logger(), "Z: %.3f (Tgt: %.3f), R/P: %.2f/%.2f, Contacts: %d, Forces: [%.1f, %.1f, %.1f, %.1f]", 
                            current_q_(2), target_q_(2), euler[0], euler[1], c_count,
                            contact_forces_[0], contact_forces_[1], contact_forces_[2], contact_forces_[3]);
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "QP Solve Failed");
        }
    }

    std::shared_ptr<PinocchioInterface> pinocchio_;
    std::unique_ptr<WbcSolver> solver_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_subs_[4];
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Eigen::VectorXd current_q_, current_v_, target_q_;
    std::vector<bool> contact_status_;
    std::vector<double> contact_forces_;
    std::map<std::string, int> joint_name_to_index_;
    bool state_received_, imu_received_, pose_received_, target_q_set_;
};

} // namespace legged_wbc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<legged_wbc::WbcNode>());
    rclcpp::shutdown();
    return 0;
}
