#include <vector>
#include <deque>
#include <Eigen/Core>

#include <ros/ros.h>
#include <camera_info_manager/camera_info_manager.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Twist.h>
#include <eigen_conversions/eigen_msg.h>

#include <h_vs/homography_2d_vs.h>
#include <h_vs/k_intrinsics.h>

// Typedefs
using RowMajorMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;


// Forward declarations
Homography2DVisualServo hvs;

ros::Subscriber G_sub;
ros::Publisher twist_pub;
ros::ServiceServer K_serv;

// Buffer for noise removal
std::deque<Eigen::VectorXd> twist_buffer;
int twist_buffer_len;


// Projective homography callback
void GCb(const std_msgs::Float64MultiArrayConstPtr G_msg) {

    // G to eigen via mapping
    Eigen::Matrix3d G = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(G_msg->data.data());

    // Compute feedback
    auto twist = hvs.computeFeedback(G);  // defaults to p_star = principal point

    // Compute moving average
    twist_buffer.push_back(twist);
    if (twist_buffer.size() > twist_buffer_len) {
        twist_buffer.pop_front();
    }

    twist.setZero();
    for (auto& twist_i: twist_buffer) {
        twist += twist_i/twist_buffer.size();
    }

    // Publish linear and angular velocity as twist
    geometry_msgs::Twist twist_msg;
    tf::twistEigenToMsg(twist, twist_msg);

    twist_pub.publish(twist_msg);
};


// Camera intrinsic service callback
bool KCb(h_vs::k_intrinsicsRequest& request, h_vs::k_intrinsicsResponse& response) {

    for (auto& dim : request.K.layout.dim) {
        if (dim.size != 3) {
            ROS_ERROR("Received camera intrinsics of wrong dimension %d for axis %s", dim.size, dim.label.c_str());
            return false;
        }
    }

    // K to Eigen via mapping
    Eigen::Matrix3d K = Eigen::Map<const RowMajorMatrix3d>(request.K.data.data());

    // Set
    hvs.K(K);

    // Generate response
    response.K.layout = request.K.layout;
    response.K.data.resize(K.size());
    Eigen::Map<RowMajorMatrix3d>(response.K.data.data()) = hvs.K();

    return true;
};


int main(int argc, char** argv) {

    ros::init(argc, argv, "h_vs_node");
    auto nh = ros::NodeHandle();

    // Read parameters
    std::string cname, url;
    std::vector<double> std_lambda_v, std_lambda_w;

    nh.getParam("h_vs_node/cname", cname);
    nh.getParam("h_vs_node/url", url);
    nh.getParam("lambda_v", std_lambda_v);
    nh.getParam("lambda_w", std_lambda_w);
    nh.getParam("twist_buffer_len", twist_buffer_len);

    camera_info_manager::CameraInfoManager camera_info(nh, cname, url);
    auto camera_matrix = camera_info.getCameraInfo().K;

    // Map parameters to eigen types
    Eigen::Vector3d lambda_v = Eigen::Vector3d::Map(std_lambda_v.data());
    Eigen::Vector3d lambda_w = Eigen::Vector3d::Map(std_lambda_w.data());
    Eigen::Matrix3d K = Eigen::Map<RowMajorMatrix3d>(camera_matrix.data());

    // Initialize visual servo
    hvs = Homography2DVisualServo(
        K, 
        lambda_v, 
        lambda_w
    );

    // Subscribe to projective homography and publish desired linear and angular velocity
    G_sub = nh.subscribe<std_msgs::Float64MultiArray>("visual_servo/G", 1, GCb);
    twist_pub = nh.advertise<geometry_msgs::Twist>("visual_servo/twist", 1);

    // Service to set camera intrinsic
    K_serv = nh.advertiseService("visual_servo/K", KCb);

    ros::spin();

    G_sub.shutdown();
    twist_pub.shutdown();

    return 0;
};
