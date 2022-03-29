#!/usr/bin/python

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo

from homography_generators import BaseHomographyGenerator

class HGenNode(Node):
    inf_sub_: rclpy.subscription.Subscription
    img_sub_: rclpy.subscription.Subscription
    hom_pub_: rclpy.publisher.Publisher
    
    def __init__(self, node_name: str="h_gen_node"):
        super().__init__(node_name=node_name)
        self.inf_sub_ = self.create_subscription(CameraInfo, "~/camera_info", self.infCb_, rclpy.qos.qos_profile_system_default)  # launch in node namespace ~/*
        self.img_sub_ = self.create_subscription(Image, "~/image_raw", self.imgCb_, rclpy.qos.qos_profile_system_default)
        self.hom_pub_ = self.create_publisher(Float64MultiArray, "~/G", rclpy.qos.qos_profile_system_default)

    def infCb_(self, msg: CameraInfo) -> None:
        self.get_logger().info("Got height {}".format(msg.height))

    def imgCb_(self, msg: Image) -> None:
        
        # self.hom_pub_.publish(response)
        return

def main(args=None):
    rclpy.init(args=args)
    h_gen_node = HGenNode("h_gen_node")
    rclpy.spin(h_gen_node)
    h_gen_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
