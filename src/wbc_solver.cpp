#include "legged_wbc/wbc_solver.hpp"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <iostream>
#include <Eigen/Geometry>

namespace legged_wbc {

WbcSolver::WbcSolver(std::shared_ptr<PinocchioInterface> pinocchio)
    : pinocchio_(pinocchio),
      qp_solver_(n_vars, n_constraints) {
    
    for (const auto& name : joint_names_) {
        joint_v_indices_.push_back(pinocchio_->getJointIdxV(pinocchio_->getJointId(name)));
    }
    
    weight_qdd_ = Eigen::VectorXd::Zero(nv);
    weight_qdd_.head(6).setConstant(0.01); // Very low penalty for base movement
    weight_qdd_.tail(10).setConstant(0.1); // Joint qdd penalty
    weight_f_ = Eigen::VectorXd::Constant(nf, 1e-4);
    weight_contact_force_ = 0.0;
    weight_contact_ = 0.0;
    weight_com_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    weight_base_linear_ = 1.0;
    weight_base_angular_ = 1.0;
    base_xy_kp_ = 40.0;
    base_xy_kd_ = 4.0;
    base_height_kp_ = 20.0;
    base_height_kd_ = 3.0;
    base_angular_kp_ = 20.0;
    base_angular_kd_ = 3.0;
    com_target_x_ = 0.02;
    torque_limits_ = Eigen::VectorXd::Zero(10);
    torque_limits_ << 28.0, 60.0, 60.0, 60.0, 28.0,
                      28.0, 60.0, 60.0, 60.0, 28.0;
    
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp_solver_.setOptions(options);

    contact_target_pos_.resize(contact_frame_names_.size(), Eigen::Vector3d::Zero());
    contact_active_.resize(contact_frame_names_.size(), false);
}

void WbcSolver::setContactForceWeight(double weight) {
    weight_contact_force_ = std::max(0.0, weight);
}

void WbcSolver::setContactTaskWeight(double weight) {
    weight_contact_ = std::max(0.0, weight);
}

void WbcSolver::setBaseAngularGains(double kp, double kd) {
    base_angular_kp_ = std::max(0.0, kp);
    base_angular_kd_ = std::max(0.0, kd);
}

void WbcSolver::setComTargetX(double x) {
    com_target_x_ = x;
}

void WbcSolver::setComWeight(double weight) {
    double clamped = std::max(0.0, weight);
    weight_com_ = Eigen::Vector3d(clamped, clamped, clamped);
}

WbcResult WbcSolver::solve(const Eigen::VectorXd& q,
                           const Eigen::VectorXd& v,
                           const Eigen::VectorXd& q_target,
                           const std::vector<bool>& contact_status) {
    auto& model = pinocchio_->getModel();
    auto& data = pinocchio_->getData();
    
    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(model, data, q, v);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::centerOfMass(model, data, q, v);
    pinocchio::jacobianCenterOfMass(model, data, q);

    Eigen::VectorXd qdd_ref = Eigen::VectorXd::Zero(nv);
    // Posture task (joints only)
    for (int i = 0; i < 10; ++i) {
        int idx_q = pinocchio_->getJointIdxQ(pinocchio_->getJointId(joint_names_[i]));
        int idx_v = pinocchio_->getJointIdxV(pinocchio_->getJointId(joint_names_[i]));
        qdd_ref(idx_v) = 150.0 * (q_target(idx_q) - q(idx_q)) - 15.0 * v(idx_v);
    }
    
    Eigen::MatrixXd H_eig = Eigen::MatrixXd::Zero(n_vars, n_vars);
    Eigen::VectorXd g_eig = Eigen::VectorXd::Zero(n_vars);
    
    H_eig.block(0, 0, nv, nv).diagonal() = weight_qdd_;
    g_eig.segment(0, nv) = -weight_qdd_.array() * qdd_ref.array();
    
    // 1. CoM Task
    Eigen::Vector3d com_pos = data.com[0];
    Eigen::Vector3d com_vel = data.vcom[0];
    Eigen::MatrixXd J_com = data.Jcom;
    
    // Target CoM X slightly forward (+0.02) to compensate for battery/offset
    Eigen::Vector3d com_target = q_target.head(3);
    com_target(0) = com_target_x_; // Geometric center offset
    com_target(1) = 0.00;
    
    Eigen::Vector3d com_acc_target = 50.0 * (com_target - com_pos) - 5.0 * com_vel;
    H_eig.block(0, 0, nv, nv) += J_com.transpose() * weight_com_.asDiagonal() * J_com;
    g_eig.segment(0, nv) += J_com.transpose() * weight_com_.asDiagonal() * (-com_acc_target);

    // 2. Base Linear Accel Task (world tracking, expressed in local frame)
    Eigen::Quaterniond q_curr(q(6), q(3), q(4), q(5));
    Eigen::Vector3d v_base_local = v.head(3);
    Eigen::Vector3d v_base_world = q_curr * v_base_local;
    Eigen::Vector3d base_acc_world;
    base_acc_world.x() = base_xy_kp_ * (q_target(0) - q(0)) - base_xy_kd_ * v_base_world.x();
    base_acc_world.y() = base_xy_kp_ * (q_target(1) - q(1)) - base_xy_kd_ * v_base_world.y();
    base_acc_world.z() = base_height_kp_ * (q_target(2) - q(2)) - base_height_kd_ * v_base_world.z();
    Eigen::Vector3d base_acc_local = q_curr.inverse() * base_acc_world;
    H_eig.block(0, 0, 3, 3).diagonal().array() += weight_base_linear_;
    g_eig.segment(0, 3) += weight_base_linear_ * (-base_acc_local);

    // 3. Base Orientation Task
    Eigen::Quaterniond q_tgt(q_target(6), q_target(3), q_target(4), q_target(5));
    Eigen::Quaterniond q_error = q_tgt * q_curr.inverse(); // Global error axis
    
    // Rotate error into LOCAL base frame to match Pinocchio velocity convention
    Eigen::AngleAxisd aa_error(q_error);
    Eigen::Vector3d ori_err_world = aa_error.axis() * aa_error.angle();
    Eigen::Vector3d ori_err_local = q_curr.inverse() * ori_err_world;
    if (ori_err_local.norm() > 0.5) ori_err_local = 0.5 * ori_err_local.normalized();
    
    Eigen::Vector3d ori_acc_target = base_angular_kp_ * ori_err_local - base_angular_kd_ * v.segment(3, 3);
    
    H_eig.block(3, 3, 3, 3).diagonal().array() += weight_base_angular_;
    g_eig.segment(3, 3) += weight_base_angular_ * (-ori_acc_target);

    // 4. Contact Constraints as Soft Tasks (Kinematics)
    double weight_contact = weight_contact_;
    for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
        if (!contact_status[i]) continue; 
        
        auto frame_id = model.getFrameId(contact_frame_names_[i]);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nv);
        pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        Eigen::MatrixXd J_trans = J.block(0, 0, 3, nv);
        
        Eigen::Vector3d foot_pos = data.oMf[frame_id].translation();
        if (!contact_active_[i]) {
            contact_target_pos_[i] = foot_pos;
            contact_target_pos_[i].z() = 0.0;
            contact_active_[i] = true;
        }
        Eigen::Vector3d foot_vel = J_trans * v;
        // Keep foot fixed at contact start pose
        Eigen::Vector3d foot_qdd_target = -400.0 * (foot_pos - contact_target_pos_[i]) - 40.0 * foot_vel;
        
        H_eig.block(0, 0, nv, nv) += weight_contact * J_trans.transpose() * J_trans;
        g_eig.segment(0, nv) += weight_contact * J_trans.transpose() * (-foot_qdd_target);
    }
    for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
        if (!contact_status[i]) {
            contact_active_[i] = false;
        }
    }
    H_eig.block(nv, nv, nf, nf).diagonal() = weight_f_;
    H_eig.diagonal().array() += 1e-3; 

    // 5. Contact Force Task (stance support)
    Eigen::VectorXd f_target = Eigen::VectorXd::Zero(nf);
    Eigen::VectorXd weight_f_contact = Eigen::VectorXd::Zero(nf);
    int contact_count = 0;
    for (bool c : contact_status) {
        if (c) {
            contact_count++;
        }
    }
    if (contact_count > 0) {
        double total_mass = data.mass[0];
        double fz_per_contact = total_mass * 9.81 / static_cast<double>(contact_count);
        for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
            if (!contact_status[i]) continue;
            f_target.segment(3 * i, 3) = Eigen::Vector3d(0.0, 0.0, fz_per_contact);
            weight_f_contact.segment(3 * i, 3).setConstant(weight_contact_force_);
        }
        H_eig.block(nv, nv, nf, nf).diagonal().array() += weight_f_contact.array();
        g_eig.segment(nv, nf).array() += -weight_f_contact.array() * f_target.array();
    }

    Eigen::MatrixXd A_eig = Eigen::MatrixXd::Zero(n_constraints, n_vars);
    Eigen::VectorXd LB = Eigen::VectorXd::Zero(n_constraints);
    Eigen::VectorXd UB = Eigen::VectorXd::Zero(n_constraints);
    
    A_eig.block(0, 0, 6, nv) = data.M.block(0, 0, 6, nv);
    for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
        auto frame_id = model.getFrameId(contact_frame_names_[i]);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nv);
        pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        A_eig.block(0, nv + 3 * i, 6, 3) = -J.block(0, 0, 3, nv).transpose().block(0, 0, 6, 3);
    }
    LB.head(6) = -data.nle.head(6);
    UB.head(6) = -data.nle.head(6);
    
    double mu = 0.6;
    for (int i = 0; i < 4; ++i) {
        int off_c = 6 + 5 * i;
        int off_f = nv + 3 * i;
        if (contact_status[i]) {
            A_eig(off_c, off_f + 2) = 1.0; LB(off_c) = 0.0; UB(off_c) = 500.0;
            A_eig(off_c+1, off_f) = 1.0; A_eig(off_c+1, off_f+2) = -mu; LB(off_c+1) = -1e10; UB(off_c+1) = 0.0;
            A_eig(off_c+2, off_f) = -1.0; A_eig(off_c+2, off_f+2) = -mu; LB(off_c+2) = -1e10; UB(off_c+2) = 0.0;
            A_eig(off_c+3, off_f+1) = 1.0; A_eig(off_c+3, off_f+2) = -mu; LB(off_c+3) = -1e10; UB(off_c+3) = 0.0;
            A_eig(off_c+4, off_f+1) = -1.0; A_eig(off_c+4, off_f+2) = -mu; LB(off_c+4) = -1e10; UB(off_c+4) = 0.0;
        } else {
            A_eig(off_c, off_f + 2) = 1.0; LB(off_c) = 0.0; UB(off_c) = 0.0;
            A_eig(off_c+1, off_f) = 1.0; LB(off_c+1) = 0.0; UB(off_c+1) = 0.0;
            A_eig(off_c+2, off_f) = 1.0; LB(off_c+2) = 0.0; UB(off_c+2) = 0.0;
            A_eig(off_c+3, off_f+1) = 1.0; LB(off_c+3) = 0.0; UB(off_c+3) = 0.0;
            A_eig(off_c+4, off_f+1) = 1.0; LB(off_c+4) = 0.0; UB(off_c+4) = 0.0;
        }
    }

    // Torque limits task: tau_min <= M*qdd + nle - J^T f <= tau_max
    int off_tau = 26;
    for (int i = 0; i < 10; ++i) {
        int idx_v = joint_v_indices_[i];
        A_eig.block(off_tau + i, 0, 1, nv) = data.M.row(idx_v);
        for (size_t c = 0; c < contact_frame_names_.size(); ++c) {
            auto frame_id = model.getFrameId(contact_frame_names_[c]);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nv);
            pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
            Eigen::RowVectorXd jt_row = J.block(0, 0, 3, nv).transpose().row(idx_v);
            A_eig.block(off_tau + i, nv + 3 * c, 1, 3) = -jt_row;
        }
        LB(off_tau + i) = -torque_limits_(i) - data.nle(idx_v);
        UB(off_tau + i) = torque_limits_(i) - data.nle(idx_v);
    }
    
    Eigen::Matrix<double, n_vars, n_vars, Eigen::RowMajor> H_rm = H_eig;
    Eigen::Matrix<double, n_constraints, n_vars, Eigen::RowMajor> A_rm = A_eig;
    
    int nWSR = 1000;
    auto status = qp_solver_.init(H_rm.data(), g_eig.data(), A_rm.data(), nullptr, nullptr, LB.data(), UB.data(), nWSR);
    
    if (status == qpOASES::SUCCESSFUL_RETURN) {
        Eigen::VectorXd x_sol = Eigen::VectorXd::Zero(n_vars);
        qp_solver_.getPrimalSolution(x_sol.data());
        WbcResult res;
        res.success = true;
        res.qdd = x_sol.head(nv);
        res.f = x_sol.tail(nf);
        
        Eigen::VectorXd tau_full = data.M * res.qdd + data.nle;
        for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
            auto frame_id = model.getFrameId(contact_frame_names_[i]);
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nv);
            pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
            tau_full -= J.transpose() * (Eigen::VectorXd(6) << res.f.segment(3 * i, 3), Eigen::Vector3d::Zero()).finished();
        }
        res.tau.resize(10);
        for (int i = 0; i < 10; ++i) res.tau(i) = tau_full(joint_v_indices_[i]);
        return res;
    } else {
        WbcResult res;
        res.success = false;
        return res;
    }
}

} // namespace legged_wbc
