#include <vector>
#include <Eigen/Core>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Twist.h>
#include <eigen_conversions/eigen_msg.h>

#include <h_vs/homography_2d_vs.h>


// Forward declarations
Homography2DVisualServo hvs;

ros::Subscriber G_sub;
ros::Publisher dtwist_pub;


// Projective homography callback
void GCb(const std_msgs::Float64MultiArrayConstPtr G_msg) {

    // G to eigen via mapping
    Eigen::Matrix3d G = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(G_msg->data.data());

    // Compute feedback
    auto dtwist = hvs.computeFeedback(G);  // defaults to p_star = principal point

    // Publish linear and angular velocity as twist
    geometry_msgs::Twist dtwist_msg;
    tf::twistEigenToMsg(dtwist, dtwist_msg);

    dtwist_pub.publish(dtwist_msg);
};


int main(int argc, char** argv) {

    ros::init(argc, argv, "visual_servo");
    auto nh = ros::NodeHandle();

    // Read parameters
    std::vector<double> std_lambda_v, std_lambda_w, camera_matrix;

    nh.getParam("lambda_v", std_lambda_v);
    nh.getParam("lambda_w", std_lambda_w);
    nh.getParam("camera_matrix/data", camera_matrix);

    // Map parameters to eigen types
    Eigen::Vector3d lambda_v = Eigen::Vector3d::Map(std_lambda_v.data());
    Eigen::Vector3d lambda_w = Eigen::Vector3d::Map(std_lambda_w.data());
    Eigen::Matrix3d K = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(camera_matrix.data());

    // Initialize visual servo
    hvs = Homography2DVisualServo(
        K, 
        lambda_v, 
        lambda_w
    );

    // Subscribe to projective homography and publish desired linear and angular velocity
    G_sub = nh.subscribe<std_msgs::Float64MultiArray>("visual_servo/G", 1, GCb);
    dtwist_pub = nh.advertise<geometry_msgs::Twist>("visual_servo/dtwist", 1);

    ros::spin();
};
