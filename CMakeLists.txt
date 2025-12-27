cmake_minimum_required(VERSION 3.8)
project(legged_wbc)

set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(qpOASES REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${pinocchio_INCLUDE_DIRS}
  ${qpOASES_INCLUDE_DIR}
)

# Pinocchio Interface Library
add_library(pinocchio_interface SHARED
  src/pinocchio_interface.cpp
)
target_compile_options(pinocchio_interface PRIVATE -w)
ament_target_dependencies(pinocchio_interface rclcpp pinocchio)

# WBC Solver Library
add_library(wbc_solver SHARED
  src/wbc_solver.cpp
)
target_compile_options(wbc_solver PRIVATE -w)
ament_target_dependencies(wbc_solver rclcpp pinocchio)
target_link_libraries(wbc_solver 
    pinocchio_interface 
    ${qpOASES_LIBRARY}
)

# WBC Node
add_executable(wbc_node
  src/wbc_node.cpp
)
target_compile_options(wbc_node PRIVATE -w)
ament_target_dependencies(wbc_node rclcpp std_msgs sensor_msgs pinocchio)
target_link_libraries(wbc_node 
    wbc_solver 
    pinocchio_interface
    ${qpOASES_LIBRARY}
)

# Install
install(TARGETS pinocchio_interface wbc_solver ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS wbc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()
