#include "legged_wbc/pinocchio_interface.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <iostream>

namespace legged_wbc {

PinocchioInterface::PinocchioInterface() : is_loaded_(false) {}

bool PinocchioInterface::loadURDF(const std::string& urdf_path) {
    try {
        // Load as floating base
        pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model_);
        data_ = pinocchio::Data(model_);
        is_loaded_ = true;
        
        std::cout << "Loaded URDF model with " << model_.nv << " DOFs (Floating Base)" << std::endl;
        std::cout << "Gravity vector: " << model_.gravity.linear().transpose() << std::endl;
        double total_mass = 0;
        for (const auto& inertia : model_.inertias) total_mass += inertia.mass();
        std::cout << "Total Model Mass: " << total_mass << " kg" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load URDF: " << e.what() << std::endl;
        return false;
    }
}

void PinocchioInterface::update(const Eigen::VectorXd& q, const Eigen::VectorXd& v) {
    if (!is_loaded_) throw std::runtime_error("Model not loaded");
    pinocchio::computeAllTerms(model_, data_, q, v);
    pinocchio::updateFramePlacements(model_, data_);
}

Eigen::VectorXd PinocchioInterface::computeGravityTorques(const Eigen::VectorXd& q) {
    if (!is_loaded_) throw std::runtime_error("Model not loaded");
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
    Eigen::VectorXd a = Eigen::VectorXd::Zero(model_.nv);
    return pinocchio::rnea(model_, data_, q, v, a);
}

Eigen::VectorXd PinocchioInterface::computeInverseDynamics(
    const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& a) {
    if (!is_loaded_) throw std::runtime_error("Model not loaded");
    return pinocchio::rnea(model_, data_, q, v, a);
}

Eigen::VectorXd PinocchioInterface::getNonlinearEffects() const {
    if (!is_loaded_) throw std::runtime_error("Model not loaded");
    return data_.nle;
}

} // namespace legged_wbc
