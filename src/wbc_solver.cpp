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
      qp_solver_(n_vars, 26) {
    
    for (const auto& name : joint_names_) {
        joint_v_indices_.push_back(pinocchio_->getJointIdxV(pinocchio_->getJointId(name)));
    }
    
    weight_qdd_ = Eigen::VectorXd::Zero(nv);
    weight_qdd_.head(6).setConstant(0.01); // Very low penalty for base movement
    weight_qdd_.tail(10).setConstant(0.1); // Joint qdd penalty
    weight_f_ = Eigen::VectorXd::Constant(nf, 1e-4);
    weight_com_ = Eigen::Vector3d(2000.0, 2000.0, 1000.0); // High weight for CoM X,Y, lower for Z
    
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp_solver_.setOptions(options);
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
    com_target(0) = 0.02; // Geometric center offset
    com_target(1) = 0.00;
    
    Eigen::Vector3d com_acc_target = 250.0 * (com_target - com_pos) - 25.0 * com_vel;
    H_eig.block(0, 0, nv, nv) += J_com.transpose() * weight_com_.asDiagonal() * J_com;
    g_eig.segment(0, nv) += J_com.transpose() * weight_com_.asDiagonal() * (-com_acc_target);

    // 2. Base Orientation Task
    double weight_ori = 5000.0;
    Eigen::Quaterniond q_curr(q(6), q(3), q(4), q(5));
    Eigen::Quaterniond q_tgt(q_target(6), q_target(3), q_target(4), q_target(5));
    Eigen::Quaterniond q_error = q_tgt * q_curr.inverse(); // Global error axis
    
    // Rotate error into LOCAL base frame to match Pinocchio velocity convention
    Eigen::AngleAxisd aa_error(q_error);
    Eigen::Vector3d ori_err_world = aa_error.axis() * aa_error.angle();
    Eigen::Vector3d ori_err_local = q_curr.inverse() * ori_err_world;
    if (ori_err_local.norm() > 0.5) ori_err_local = 0.5 * ori_err_local.normalized();
    
    Eigen::Vector3d ori_acc_target = 250.0 * ori_err_local - 30.0 * v.segment(3, 3);
    
    H_eig.block(3, 3, 3, 3).diagonal().array() += weight_ori;
    g_eig.segment(3, 3) += weight_ori * (-ori_acc_target);

    // 3. Contact Constraints as Soft Tasks (Kinematics)
    double weight_contact = 5000.0;
    for (size_t i = 0; i < contact_frame_names_.size(); ++i) {
        if (!contact_status[i]) continue; 
        
        auto frame_id = model.getFrameId(contact_frame_names_[i]);
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, nv);
        pinocchio::getFrameJacobian(model, data, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        Eigen::MatrixXd J_trans = J.block(0, 0, 3, nv);
        
        Eigen::Vector3d foot_pos = data.oMf[frame_id].translation();
        Eigen::Vector3d foot_vel = J_trans * v;
        // Keep foot fixed on ground
        Eigen::Vector3d foot_qdd_target = -400.0 * (foot_pos - Eigen::Vector3d(foot_pos.x(), foot_pos.y(), 0.0)) - 40.0 * foot_vel;
        
        H_eig.block(0, 0, nv, nv) += weight_contact * J_trans.transpose() * J_trans;
        g_eig.segment(0, nv) += weight_contact * J_trans.transpose() * (-foot_qdd_target);
    }
    H_eig.block(nv, nv, nf, nf).diagonal() = weight_f_;
    H_eig.diagonal().array() += 1e-3; 

    Eigen::MatrixXd A_eig = Eigen::MatrixXd::Zero(26, n_vars);
    Eigen::VectorXd LB = Eigen::VectorXd::Zero(26);
    Eigen::VectorXd UB = Eigen::VectorXd::Zero(26);
    
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
            A_eig(off_c, off_f + 2) = 1.0; LB(off_c) = 1.0; UB(off_c) = 500.0;
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
    
    Eigen::Matrix<double, n_vars, n_vars, Eigen::RowMajor> H_rm = H_eig;
    Eigen::Matrix<double, 26, n_vars, Eigen::RowMajor> A_rm = A_eig;
    
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
