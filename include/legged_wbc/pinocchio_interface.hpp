#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

namespace legged_wbc {

class PinocchioInterface {
public:
    PinocchioInterface();
    ~PinocchioInterface() = default;

    bool loadURDF(const std::string& urdf_path);
    void update(const Eigen::VectorXd& q, const Eigen::VectorXd& v);
    
    Eigen::VectorXd computeGravityTorques(const Eigen::VectorXd& q);
    Eigen::VectorXd computeInverseDynamics(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& a);
    
    Eigen::VectorXd getNonlinearEffects() const;
    
    // Joint mapping helpers
    int getJointId(const std::string& name) const { return model_.getJointId(name); }
    int getJointIdxQ(int joint_id) const { return model_.joints[joint_id].idx_q(); }
    int getJointIdxV(int joint_id) const { return model_.joints[joint_id].idx_v(); }
    
    const pinocchio::Model& getModel() const { return model_; }
    pinocchio::Data& getData() { return data_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    bool is_loaded_;
};

} // namespace legged_wbc
