#include <rclcpp/rclcpp.hpp>
#include <controller_manager_msgs/srv/list_controllers.hpp>
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
#include <atomic>
#include <thread>
#include <future>
#include <Eigen/Geometry>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <array>

namespace legged_wbc {

class WbcNode : public rclcpp::Node {
public:
    WbcNode() : Node("wbc_node"), state_received_(false), imu_received_(false), pose_received_(false), joint_state_valid_(false) {
        this->declare_parameter("urdf_path", "/tmp/hunter_generated.urdf");
        this->declare_parameter("control_frequency", 500.0);
        this->declare_parameter("force_stance_contacts", true);
        this->declare_parameter("target_height", 0.0);
        this->declare_parameter("wait_for_controller", true);
        this->declare_parameter("controller_name", "forward_command_controller");
        this->declare_parameter("require_joint_velocity", true);
        this->declare_parameter("debug_fall_log", false);
        this->declare_parameter("fall_log_roll_pitch_threshold", 0.5);
        this->declare_parameter("fall_log_z_threshold", 0.4);
        this->declare_parameter("task_file", "");
        this->declare_parameter("reference_file", "");
        this->declare_parameter("weight_contact_force", -1.0);
        this->declare_parameter("weight_contact", 0.0);
        this->declare_parameter("base_angular_kp", -1.0);
        this->declare_parameter("base_angular_kd", -1.0);
        this->declare_parameter("com_target_x", 0.02);
        this->declare_parameter("weight_com", -1.0);
        
        std::string urdf_path = this->get_parameter("urdf_path").as_string();
        double freq = this->get_parameter("control_frequency").as_double();
        force_stance_contacts_ = this->get_parameter("force_stance_contacts").as_bool();
        target_height_ = this->get_parameter("target_height").as_double();
        wait_for_controller_ = this->get_parameter("wait_for_controller").as_bool();
        controller_name_ = this->get_parameter("controller_name").as_string();
        require_joint_velocity_ = this->get_parameter("require_joint_velocity").as_bool();
        debug_fall_log_ = this->get_parameter("debug_fall_log").as_bool();
        fall_log_roll_pitch_threshold_ =
            this->get_parameter("fall_log_roll_pitch_threshold").as_double();
        fall_log_z_threshold_ = this->get_parameter("fall_log_z_threshold").as_double();
        task_file_ = this->get_parameter("task_file").as_string();
        reference_file_ = this->get_parameter("reference_file").as_string();
        weight_contact_force_ = this->get_parameter("weight_contact_force").as_double();
        weight_contact_ = this->get_parameter("weight_contact").as_double();
        base_angular_kp_ = this->get_parameter("base_angular_kp").as_double();
        base_angular_kd_ = this->get_parameter("base_angular_kd").as_double();
        com_target_x_ = this->get_parameter("com_target_x").as_double();
        weight_com_ = this->get_parameter("weight_com").as_double();
        
        pinocchio_ = std::make_shared<PinocchioInterface>();
        if (!pinocchio_->loadURDF(urdf_path)) {
            throw std::runtime_error("Failed to load URDF");
        }
        
        solver_ = std::make_unique<WbcSolver>(pinocchio_);
        applyTaskSettings();
        loadDefaultJointState();
        
        int nq = pinocchio_->getModel().nq;
        int nv = pinocchio_->getModel().nv;
        current_q_ = Eigen::VectorXd::Zero(nq);
        current_v_ = Eigen::VectorXd::Zero(nv);
        current_q_(6) = 1.0; 
        
        target_q_ = Eigen::VectorXd::Zero(nq);
        target_q_(6) = 1.0;
        target_q_set_ = false;
        target_z_ = 0.0;
        
        contact_status_.resize(4, false);
        contact_forces_.resize(4, 0.0);
        contact_raw_forces_.resize(4, 0.0);

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&WbcNode::jointStateCallback, this, std::placeholders::_1));
        
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu_imu_sensor/imu", 10, std::bind(&WbcNode::imuCallback, this, std::placeholders::_1));
            
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/base_link_pose_sensor/pose", 10, std::bind(&WbcNode::poseCallback, this, std::placeholders::_1));
            
        setupWrenchSubscriptions();
        
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/forward_command_controller/commands", 10);

        if (wait_for_controller_) {
            controller_client_ = this->create_client<controller_manager_msgs::srv::ListControllers>(
                "/controller_manager/list_controllers");
            controller_wait_thread_ = std::thread(&WbcNode::waitForController, this);
        } else {
            controller_ready_.store(true);
        }
        
        auto period = std::chrono::duration<double>(1.0 / freq);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&WbcNode::controlLoop, this));
            
        RCLCPP_INFO(this->get_logger(), "WBC Node Started (Contact-Aware)");
    }

    ~WbcNode() override {
        if (controller_wait_thread_.joinable()) {
            controller_wait_thread_.join();
        }
    }

private:
    void applyTaskSettings() {
        if (!task_file_.empty()) {
            try {
                boost::property_tree::ptree pt;
                boost::property_tree::read_info(task_file_, pt);
                if (weight_contact_force_ < 0.0) {
                    auto opt = pt.get_optional<double>("weight.contactForce");
                    if (opt) {
                        weight_contact_force_ = opt.value();
                    }
                }
                if (base_angular_kp_ < 0.0) {
                    auto opt = pt.get_optional<double>("baseAngularTask.kp");
                    if (opt) {
                        base_angular_kp_ = opt.value();
                    }
                }
                if (base_angular_kd_ < 0.0) {
                    auto opt = pt.get_optional<double>("baseAngularTask.kd");
                    if (opt) {
                        base_angular_kd_ = opt.value();
                    }
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(),
                    "Failed to read task_file '%s': %s",
                    task_file_.c_str(), e.what());
            }
        }
        if (weight_contact_force_ < 0.0) {
            weight_contact_force_ = 0.0;
        }
        solver_->setContactForceWeight(weight_contact_force_);
        solver_->setContactTaskWeight(weight_contact_);
        if (base_angular_kp_ < 0.0) {
            base_angular_kp_ = 0.0;
        }
        if (base_angular_kd_ < 0.0) {
            base_angular_kd_ = 0.0;
        }
        solver_->setBaseAngularGains(base_angular_kp_, base_angular_kd_);
        solver_->setComTargetX(com_target_x_);
        if (weight_com_ < 0.0) {
            weight_com_ = 0.0;
        }
        solver_->setComWeight(weight_com_);
        RCLCPP_INFO(this->get_logger(),
            "WBC weights: contact_force=%.3f contact_task=%.3f base_angular_kp=%.2f base_angular_kd=%.2f com_target_x=%.3f com_weight=%.3f",
            weight_contact_force_, weight_contact_, base_angular_kp_, base_angular_kd_, com_target_x_, weight_com_);
    }
    void setupWrenchSubscriptions() {
        std::string frames[] = {"leg_l_f1_site", "leg_l_f2_site", "leg_r_f1_site", "leg_r_f2_site"};
        for (int i = 0; i < 4; ++i) {
            std::string topic = "/" + frames[i] + "_wrench_sensor/wrench";
            wrench_subs_[i] = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                topic, 10, [this, i](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
                    this->contact_raw_forces_[i] = std::abs(msg->wrench.force.z); // use abs for different orientations
                });
        }
    }

    void waitForController() {
        auto start = std::chrono::steady_clock::now();
        while (rclcpp::ok()) {
            if (!controller_client_->wait_for_service(std::chrono::seconds(1))) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for /controller_manager/list_controllers...");
                continue;
            }
            auto request = std::make_shared<controller_manager_msgs::srv::ListControllers::Request>();
            auto future = controller_client_->async_send_request(request);
            if (future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
                const auto response = future.get();
                for (const auto& controller : response->controller) {
                    if (controller.name == controller_name_ && controller.state == "active") {
                        controller_ready_.store(true);
                        RCLCPP_INFO(this->get_logger(),
                            "Controller '%s' is active. WBC control enabled.",
                            controller_name_.c_str());
                        return;
                    }
                }
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Controller '%s' not active yet. Waiting...", controller_name_.c_str());
            }
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
                RCLCPP_WARN(this->get_logger(),
                    "Controller '%s' not active yet. WBC will keep waiting.",
                    controller_name_.c_str());
                start = std::chrono::steady_clock::now();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        static rclcpp::Time last_time = this->get_clock()->now();
        static Eigen::Vector3d last_pos(0,0,0);
        static Eigen::Quaterniond last_q(1.0, 0.0, 0.0, 0.0);
        
        rclcpp::Time current_time = this->get_clock()->now();
        double dt = (current_time - last_time).seconds();

        Eigen::Quaterniond q_msg(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);
        if (last_q.dot(q_msg) < 0.0) {
            q_msg.coeffs() *= -1.0;
        }
        q_msg.normalize();
        last_q = q_msg;

        current_q_(0) = msg->pose.position.x;
        current_q_(1) = msg->pose.position.y;
        current_q_(2) = msg->pose.position.z;
        current_q_(3) = q_msg.x();
        current_q_(4) = q_msg.y();
        current_q_(5) = q_msg.z();
        current_q_(6) = q_msg.w();

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

        const bool has_position = msg->position.size() >= msg->name.size();
        const bool has_velocity = msg->velocity.size() >= msg->name.size();
        if (!has_position || (require_joint_velocity_ && !has_velocity)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "JointState arrays too small: name=%zu position=%zu velocity=%zu",
                msg->name.size(), msg->position.size(), msg->velocity.size());
            joint_state_valid_ = false;
            return;
        }

        for (size_t i = 0; i < robot_joints_.size(); ++i) {
            auto it = joint_name_to_index_.find(robot_joints_[i]);
            if (it != joint_name_to_index_.end()) {
                int pin_id = pinocchio_->getJointId(robot_joints_[i]);
                if (pin_id >= 0) {
                    current_q_(pinocchio_->getJointIdxQ(pin_id)) = msg->position[it->second];
                    double vel = 0.0;
                    if (it->second < msg->velocity.size()) {
                        vel = msg->velocity[it->second];
                    }
                    current_v_(pinocchio_->getJointIdxV(pin_id)) = vel;
                }
            }
        }
        
        if (!target_q_set_ && pose_received_ && (!wait_for_controller_ || controller_ready_.load())) {
            target_q_ = current_q_;
            target_q_(3) = 0.0; target_q_(4) = 0.0; target_q_(5) = 0.0; target_q_(6) = 1.0;
            if (default_joint_state_loaded_) {
                for (size_t i = 0; i < robot_joints_.size(); ++i) {
                    int pin_id = pinocchio_->getJointId(robot_joints_[i]);
                    if (pin_id >= 0 && i < static_cast<size_t>(default_joint_state_.size())) {
                        target_q_(pinocchio_->getJointIdxQ(pin_id)) = default_joint_state_(static_cast<int>(i));
                    }
                }
            }
            if (target_height_ > 0.0) {
                target_z_ = target_height_;
                target_q_(2) = target_height_;
            } else {
                target_z_ = target_q_(2);
            }
            target_q_set_ = true;
            RCLCPP_INFO(this->get_logger(), "WBC: Target Initialized at height %.3f", current_q_(2));
            if (default_joint_state_loaded_) {
                std::ostringstream joints_stream;
                joints_stream << "WBC target_q_ joints:";
                for (size_t i = 0; i < robot_joints_.size(); ++i) {
                    int pin_id = pinocchio_->getJointId(robot_joints_[i]);
                    if (pin_id >= 0) {
                        joints_stream << " " << robot_joints_[i] << "="
                                      << target_q_(pinocchio_->getJointIdxQ(pin_id));
                    }
                }
                RCLCPP_INFO(this->get_logger(), "%s", joints_stream.str().c_str());
            }
        }
        joint_state_valid_ = true;
        state_received_ = true;
    }

    void controlLoop() {
        if (wait_for_controller_ && !controller_ready_.load()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Controller '%s' not active yet. Skipping WBC control.",
                controller_name_.c_str());
            return;
        }
        if (!state_received_ || !joint_state_valid_ || !imu_received_ || !pose_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for state/imu/pose: state=%d joint_state_valid=%d imu=%d pose=%d",
                state_received_, joint_state_valid_, imu_received_, pose_received_);
            return;
        }

        // Hold the initial height target to avoid squat-induced instability.
        if (target_q_set_) {
            target_q_(2) = target_z_;
        }

        pinocchio_->update(current_q_, current_v_);

        // Contact heuristic with hysteresis for stability.
        auto& model = pinocchio_->getModel();
        auto& data = pinocchio_->getData();
        double foot_z[4] = {0.0, 0.0, 0.0, 0.0};
        const std::string frame_names[] = {"leg_l_f1_link", "leg_l_f2_link", "leg_r_f1_link", "leg_r_f2_link"};
        for (int i = 0; i < 4; ++i) {
            auto frame_id = model.getFrameId(frame_names[i]);
            if (frame_id == model.nframes) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Foot frame not found: %s", frame_names[i].c_str());
                contact_status_[i] = false;
                contact_forces_[i] = 0.0;
                foot_z[i] = 0.0;
                continue;
            }
            foot_z[i] = data.oMf[frame_id].translation().z();
            double force_z = contact_raw_forces_[i];
            if (!contact_status_[i]) {
                if (force_z > 5.0 || foot_z[i] < 0.01) {
                    contact_status_[i] = true;
                }
            } else {
                if (force_z < 1.0 && foot_z[i] > 0.03) {
                    contact_status_[i] = false;
                }
            }
            contact_forces_[i] = force_z;
        }
        if (force_stance_contacts_ && target_q_set_) {
            for (int i = 0; i < 4; ++i) {
                contact_status_[i] = true;
            }
        }
        auto result = solver_->solve(current_q_, current_v_, target_q_, contact_status_);
        if (result.success) {
            auto cmd = std_msgs::msg::Float64MultiArray();
            cmd.data.resize(10);
            int sat_count = 0;
            double max_abs_tau = 0.0;
            for (int i = 0; i < 10; ++i) {
                max_abs_tau = std::max(max_abs_tau, std::abs(result.tau(i)));
                if (std::abs(result.tau(i)) > 300.0) {
                    sat_count++;
                }
                cmd.data[i] = std::clamp(result.tau(i), -300.0, 300.0);
            }
            effort_pub_->publish(cmd);
            
            Eigen::Quaterniond q(current_q_(6), current_q_(3), current_q_(4), current_q_(5));
            auto euler = q.toRotationMatrix().eulerAngles(0, 1, 2);
            static int count = 0;
            if (count++ % 100 == 0) {
                int c_count = 0;
                for(bool b : contact_status_) if(b) c_count++;
                double fz_cmd[4] = {0.0, 0.0, 0.0, 0.0};
                if (result.f.size() >= 12) {
                    for (int i = 0; i < 4; ++i) fz_cmd[i] = result.f(3 * i + 2);
                }
                RCLCPP_INFO(this->get_logger(),
                            "Z: %.3f (Tgt: %.3f), R/P: %.2f/%.2f, Contacts: %d, Forces: [%.1f, %.1f, %.1f, %.1f], CmdFz: [%.1f, %.1f, %.1f, %.1f], Sat: %d, MaxTau: %.1f",
                            current_q_(2), target_q_(2), euler[0], euler[1], c_count,
                            contact_forces_[0], contact_forces_[1], contact_forces_[2], contact_forces_[3],
                            fz_cmd[0], fz_cmd[1], fz_cmd[2], fz_cmd[3],
                            sat_count, max_abs_tau);
            }
            if (debug_fall_log_) {
                const bool unstable_rp = std::abs(euler[0]) > fall_log_roll_pitch_threshold_ ||
                                         std::abs(euler[1]) > fall_log_roll_pitch_threshold_;
                const bool low_z = current_q_(2) < fall_log_z_threshold_;
                if (unstable_rp || low_z) {
                    double fz_cmd[4] = {0.0, 0.0, 0.0, 0.0};
                    if (result.f.size() >= 12) {
                        for (int i = 0; i < 4; ++i) fz_cmd[i] = result.f(3 * i + 2);
                    }
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                        "Instability: Z %.3f (Tgt %.3f) R/P %.2f/%.2f vxyz %.2f/%.2f/%.2f "
                        "foot_z [%.3f, %.3f, %.3f, %.3f] raw_fz [%.1f, %.1f, %.1f, %.1f] "
                        "cmd_fz [%.1f, %.1f, %.1f, %.1f] max_tau %.1f",
                        current_q_(2), target_q_(2), euler[0], euler[1],
                        current_v_(0), current_v_(1), current_v_(2),
                        foot_z[0], foot_z[1], foot_z[2], foot_z[3],
                        contact_forces_[0], contact_forces_[1], contact_forces_[2], contact_forces_[3],
                        fz_cmd[0], fz_cmd[1], fz_cmd[2], fz_cmd[3],
                        max_abs_tau);
                }
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "QP Solve Failed");
        }
    }

    void loadDefaultJointState() {
        default_joint_state_loaded_ = false;
        if (reference_file_.empty()) {
            return;
        }
        default_joint_state_ = Eigen::VectorXd::Zero(robot_joints_.size());
        try {
            boost::property_tree::ptree pt;
            boost::property_tree::read_info(reference_file_, pt);
            const double scaling = pt.get<double>("defaultJointState.scaling", 1.0);
            const double default_value = pt.get<double>("defaultJointState.default", 0.0);
            size_t num_failed = 0;
            for (size_t i = 0; i < robot_joints_.size(); ++i) {
                double value = default_value;
                try {
                    value = pt.get<double>("defaultJointState.(" + std::to_string(i) + ",0)");
                } catch (const std::exception&) {
                    num_failed++;
                }
                default_joint_state_(static_cast<int>(i)) = scaling * value;
            }
            if (num_failed == robot_joints_.size()) {
                RCLCPP_WARN(this->get_logger(),
                    "reference_file '%s' did not provide defaultJointState entries.",
                    reference_file_.c_str());
                return;
            }
            if (num_failed > 0) {
                RCLCPP_WARN(this->get_logger(),
                    "reference_file '%s' missing %zu defaultJointState entries, using defaults.",
                    reference_file_.c_str(), num_failed);
            }
            default_joint_state_loaded_ = true;
            RCLCPP_INFO(this->get_logger(),
                "Loaded defaultJointState from '%s' for WBC initial pose.",
                reference_file_.c_str());
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(),
                "Failed to read reference_file '%s': %s",
                reference_file_.c_str(), e.what());
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
    rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr controller_client_;
    std::thread controller_wait_thread_;
    Eigen::VectorXd current_q_, current_v_, target_q_;
    std::vector<bool> contact_status_;
    std::vector<double> contact_forces_;
    std::vector<double> contact_raw_forces_;
    std::map<std::string, int> joint_name_to_index_;
    double target_z_;
    double target_height_;
    bool force_stance_contacts_;
    std::string task_file_;
    double weight_contact_force_;
    double weight_contact_;
    double base_angular_kp_;
    double base_angular_kd_;
    double com_target_x_;
    double weight_com_;
    bool state_received_, imu_received_, pose_received_, target_q_set_, joint_state_valid_;
    bool wait_for_controller_;
    bool require_joint_velocity_;
    bool debug_fall_log_;
    double fall_log_roll_pitch_threshold_;
    double fall_log_z_threshold_;
    std::string controller_name_;
    std::string reference_file_;
    Eigen::VectorXd default_joint_state_;
    bool default_joint_state_loaded_;
    const std::array<std::string, 10> robot_joints_ = {
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"
    };
    std::atomic<bool> controller_ready_{false};
};

} // namespace legged_wbc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<legged_wbc::WbcNode>());
    rclcpp::shutdown();
    return 0;
}
