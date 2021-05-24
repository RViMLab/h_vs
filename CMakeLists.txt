cmake_minimum_required(VERSION 3.0.2)
project(h_vs)

# Catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    camera_info_manager
    rospy
    camera_info_manager_py
    actionlib
    std_msgs
    sensor_msgs
    geometry_msgs
    eigen_conversions
    message_generation
)

# Other packages
find_package(Eigen3 REQUIRED)

# Generate services in the 'srv' folder and actions in 'action' folder
add_action_files(
    FILES h_vs.action
)

add_service_files(
  FILES capture.srv k_intrinsics.srv # http://wiki.ros.org/catkin/CMakeLists.txt#msgs_srvs_actions, properly add messages, actions, services -> update rcm_msgs with exported targets
)

add_message_files(
  FILES pairwise_distance.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES std_msgs sensor_msgs actionlib_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp camera_info_manager rospy camera_info_manager_py std_msgs geometry_msgs eigen_conversions message_runtime
    INCLUDE_DIRS include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}
    DEPENDS EIGEN3
)

# Includes
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Library
add_library(h_vs src/homography_2d_vs.cpp)

# Executable
add_executable(h_vs_node src/h_vs_node.cpp)
add_dependencies(h_vs_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(h_vs_node ${catkin_LIBRARIES} h_vs)

# Install
install(
    DIRECTORY include/h_vs/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
    TARGETS h_vs_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(
  PROGRAMS python/control_gallery_node.py python/h_gen_calibration_pattern_node.py python/h_gen_endoscopy_calibration_pattern_node.py python/h_gen_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
