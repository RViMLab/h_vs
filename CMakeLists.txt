cmake_minimum_required(VERSION 3.0.2)
project(h_vs)

# Catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    eigen_conversions
)

# Other packages
find_package(Eigen3 REQUIRED)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs eigen_conversions
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
  PROGRAMS python/h_gen_node.py python/h_gen_calibration_pattern_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
