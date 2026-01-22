#ifndef LEGGED_WBC_WBC_SOLVER_HPP
#define LEGGED_WBC_WBC_SOLVER_HPP

#include "legged_wbc/pinocchio_interface.hpp"
#include <qpOASES.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace legged_wbc {

struct WbcResult {
    bool success;
    Eigen::VectorXd tau;
    Eigen::VectorXd qdd;
    Eigen::VectorXd f;
};

class WbcSolver {
public:
    WbcSolver(std::shared_ptr<PinocchioInterface> pinocchio);

    void setContactForceWeight(double weight);
    void setContactTaskWeight(double weight);
    void setBaseAngularGains(double kp, double kd);
    void setComTargetX(double x);
    void setComWeight(double weight);
    
    WbcResult solve(const Eigen::VectorXd& q,
                    const Eigen::VectorXd& v,
                    const Eigen::VectorXd& q_target,
                    const std::vector<bool>& contact_status);

private:
    std::shared_ptr<PinocchioInterface> pinocchio_;
    qpOASES::SQProblem qp_solver_;
    
    static constexpr int nv = 16;
    static constexpr int nf = 12; 
    static constexpr int n_vars = nv + nf;
    static constexpr int n_constraints = 36;
    
    std::vector<std::string> contact_frame_names_ = {
        "leg_l_f1_link", "leg_l_f2_link", "leg_r_f1_link", "leg_r_f2_link"
    };

    std::vector<std::string> joint_names_ = {
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"
    };
    
    std::vector<int> joint_v_indices_;
    Eigen::VectorXd weight_qdd_, weight_f_, weight_com_;
    Eigen::VectorXd torque_limits_;
    double weight_contact_force_;
    double weight_contact_;
    double weight_base_linear_;
    double weight_base_angular_;
    double base_xy_kp_;
    double base_xy_kd_;
    double base_height_kp_;
    double base_height_kd_;
    double base_angular_kp_;
    double base_angular_kd_;
    double com_target_x_;
    std::vector<Eigen::Vector3d> contact_target_pos_;
    std::vector<bool> contact_active_;
};

} // namespace legged_wbc

#endif
