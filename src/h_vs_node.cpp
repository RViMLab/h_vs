#include <deque>
#include <memory>
#include <string>
#include <functional>
#include <Eigen/Core>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <h_vs/h_vs.hpp>


using RowMajorMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;


class HVsNode : public rclcpp::Node {
    public:
        HVsNode(std::string node_name = "h_vs_node") : Node(node_name) {
            this->declare_parameter("lambda_v");  // https://docs.ros.org/en/foxy/Tutorials/Using-Parameters-In-A-Class-CPP.html
            this->declare_parameter("lambda_w");
            this->declare_parameter("twist_buffer_len");
            this->declare_parameter("cname");
            this->declare_parameter("url");

            std::vector<double> lambda_v, lambda_w;

            if (!get_parameter("lambda_v", lambda_v)) { std::string error = "Failed to receive gain parameter 'lambda_v'."; RCLCPP_ERROR(get_logger(), error); throw std::runtime_error(error); };
            if (!get_parameter("lambda_w", lambda_w)) { std::string error = "Failed to receive gain parameter 'lambda_w'."; RCLCPP_ERROR(get_logger(), error); throw std::runtime_error(error); };
            if (!get_parameter("twist_buffer_len", twist_buffer_len_)) { std::string error = "Failed to receive parameters 'twist_buffer_len'."; RCLCPP_ERROR(get_logger(), error); throw std::runtime_error(error); };
            if (!get_parameter("cname", cname_)) { RCLCPP_WARN(get_logger(), "Failed to receive parameter 'cname', defaulting to 'camera'."); cname_ = "camera"; };
            if (!get_parameter("url", url_)) { RCLCPP_WARN(get_logger(), "Failed to receive parameter 'url', defaulting to ''."); url_ = ""; };

            camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, cname_, url_);
            auto K = camera_info_manager_->getCameraInfo().k;

            // Map parameters to eigen types
            lambda_v_ = Eigen::Vector3d::Map(lambda_v.data());
            lambda_w_ = Eigen::Vector3d::Map(lambda_w.data());
            K_ = Eigen::Map<const RowMajorMatrix3d>(K.data());

            // Initialize homography-based visual servo
            h_vs_ = std::make_unique<HVs>(K_, lambda_v_, lambda_w_);

            G_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                "~/G",
                rclcpp::SystemDefaultsQoS(),
                std::bind(&HVsNode::GCb, this, std::placeholders::_1)    
            );

            K_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                "~/K",
                rclcpp::SystemDefaultsQoS(), [this](const std_msgs::msg::Float64MultiArray::SharedPtr K_msg) {
                    Eigen::Matrix3d K = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(K_msg->data.data());
                    this->h_vs_->K(K);
                }
            );

            twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
                "~/twist",
                rclcpp::SystemDefaultsQoS()
            );
        }

    private:
        std::unique_ptr<HVs> h_vs_;
        Eigen::Matrix3d K_;
        Eigen::Vector3d lambda_v_, lambda_w_;

        std::string cname_, url_;
        std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr G_sub_;  // projective homography G ~ KHK^(-1), with H Euclidean homography
        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr K_sub_;  // camera intrinsics
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
        // rclcpp::Client< ??

        std::deque<Eigen::VectorXd> twist_buffer_;
        std::size_t twist_buffer_len_;

        void GCb(const std_msgs::msg::Float64MultiArray::SharedPtr G_msg) {
            // G_msg to eigen via mapping
            Eigen::Matrix3d G = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(G_msg->data.data());
            
            // Compute feedback
            auto twist = h_vs_->computeFeedback(G);

            // Compute moving average
            twist_buffer_.push_back(twist);
            if (twist_buffer_.size() > twist_buffer_len_) {
                twist_buffer_.pop_front();
            }

            twist.setZero();
            for (auto& twist_i: twist_buffer_) {
                twist += twist_i/twist_buffer_.size();
            }

            // Publish twist
            geometry_msgs::msg::Twist twist_msg;
            twist_msg.linear.x = twist[0];
            twist_msg.linear.y = twist[1];
            twist_msg.linear.z = twist[2];
            twist_msg.angular.x = twist[3];
            twist_msg.angular.y = twist[4];
            twist_msg.angular.z = twist[5];
            twist_pub_->publish(twist_msg);
        };
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HVsNode>());
    rclcpp::shutdown();
    return 0;
}
