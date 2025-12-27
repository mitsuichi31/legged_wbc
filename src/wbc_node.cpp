#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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
        current_q_(2) = 0.682;
        
        target_q_ = Eigen::VectorXd::Zero(nq);
        target_q_(6) = 1.0;
        target_q_set_ = false;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&WbcNode::jointStateCallback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_imu_sensor/imu", 10, std::bind(&WbcNode::imuCallback, this, std::placeholders::_1));
            
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/base_link_pose_sensor/pose", 10, std::bind(&WbcNode::poseCallback, this, std::placeholders::_1));
        
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/forward_command_controller/commands", 10);
        
        auto period = std::chrono::duration<double>(1.0 / freq);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&WbcNode::controlLoop, this));
            
        RCLCPP_INFO(this->get_logger(), "WBC Node Started with Ground Truth Pose Feedback");
    }

private:
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_q_(0) = msg->pose.position.x;
        current_q_(1) = msg->pose.position.y;
        current_q_(2) = msg->pose.position.z;
        current_q_(3) = msg->pose.orientation.x;
        current_q_(4) = msg->pose.orientation.y;
        current_q_(5) = msg->pose.orientation.z;
        current_q_(6) = msg->pose.orientation.w;
        pose_received_ = true;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Use IMU for angular velocity (more reliable/low latency than pose derivative)
        current_v_(3) = msg->angular_velocity.x;
        current_v_(4) = msg->angular_velocity.y;
        current_v_(5) = msg->angular_velocity.z;
        imu_received_ = true;
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (joint_name_to_index_.empty()) {
            for (size_t i = 0; i < msg->name.size(); ++i) {
                joint_name_to_index_[msg->name[i]] = i;
            }
        }

        std::string robot_joints[] = {
            "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
            "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"
        };

        for (int i = 0; i < 10; ++i) {
            auto it = joint_name_to_index_.find(robot_joints[i]);
            if (it != joint_name_to_index_.end()) {
                int pin_id = pinocchio_->getJointId(robot_joints[i]);
                if (pin_id >= 0 && pin_id < (int)pinocchio_->getModel().joints.size()) {
                    current_q_(pinocchio_->getJointIdxQ(pin_id)) = msg->position[it->second];
                    current_v_(pinocchio_->getJointIdxV(pin_id)) = msg->velocity[it->second];
                }
            }
        }
        
        if (!target_q_set_ && pose_received_) {
            target_q_ = current_q_;
            // Aim for identity orientation
            target_q_(3) = 0.0; target_q_(4) = 0.0; target_q_(5) = 0.0; target_q_(6) = 1.0;
            // Aim for 0.68m height
            target_q_(2) = 0.68; 
            target_q_set_ = true;
            RCLCPP_INFO(this->get_logger(), "WBC: Target Initialized from Ground Truth. Base Z: %.3f", current_q_(2));
        }
        state_received_ = true;
    }

    void controlLoop() {
        if (!state_received_ || !imu_received_ || !pose_received_) return;

        pinocchio_->update(current_q_, current_v_);
        auto result = solver_->solve(current_q_, current_v_, target_q_);
        if (result.success) {
            auto cmd = std_msgs::msg::Float64MultiArray();
            cmd.data.resize(10);
            for (int i = 0; i < 10; ++i) cmd.data[i] = result.tau(i);
            effort_pub_->publish(cmd);
            
            static int count = 0;
            if (count++ % 100 == 0) {
                Eigen::Quaterniond q(current_q_(6), current_q_(3), current_q_(4), current_q_(5));
                auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
                RCLCPP_INFO(this->get_logger(), "Base Z: %.3f, Roll: %.2f, Pitch: %.2f", 
                            current_q_(2), euler[0], euler[1]);
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
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Eigen::VectorXd current_q_, current_v_, target_q_;
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
