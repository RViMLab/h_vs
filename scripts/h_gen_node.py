#!/usr/bin/python

import cv2
import numpy as np
import rclpy
import cv_bridge
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import crop_and_resize, warp_perspective
from rclpy.node import Node
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Wrench

from endoscopy import BoundingCircleDetector, HomographyEstimator
from endoscopy.utils import MODEL, max_rectangle_in_circle, yt_alpha_blend

from homography_generators import BaseHomographyGenerator
from homography_generators.utils.conversions import homographyToMsg


class HGenNode(Node):
    inf_sub_: rclpy.subscription.Subscription
    img_sub_: rclpy.subscription.Subscription
    hom_pub_: rclpy.publisher.Publisher
    wrench_sub_: rclpy.subscription.Subscription
    class_prob_pub_: rclpy.publisher.Publisher
    bridge_: cv_bridge.CvBridge
    circle_: BoundingCircleDetector

    def __init__(self, node_name: str="h_gen_node"):
        super().__init__(node_name=node_name)
        self.inf_sub_ = self.create_subscription(CameraInfo, "~/camera_info", self.infCb_, rclpy.qos.qos_profile_system_default)  # launch in node namespace ~/*
        self.img_sub_ = self.create_subscription(Image, "~/image_raw", self.imgCb_, rclpy.qos.qos_profile_system_default)
        self.hom_pub_ = self.create_publisher(Float64MultiArray, "~/G", rclpy.qos.qos_profile_system_default)
        self.wrench_sub_ = self.create_subscription(Wrench, "~/wrench", self.wrenchCb, rclpy.qos.qos_profile_system_default)
        self.class_prob_pub_ = self.create_publisher(Float64MultiArray, "~/class_probability", rclpy.qos.qos_profile_system_default)
        self.bridge_ = cv_bridge.CvBridge()
        self.circle_ = BoundingCircleDetector(model=MODEL.SEGMENTATION.UNET_RESNET_34, device="cuda")
        self.h_est_ = HomographyEstimator(model=MODEL.HOMOGRAPHY_ESTIMATION.RESNET_34, device="cuda")
        self.get_logger().info("loaded model")

        self.prev_crp_ = None

    def infCb_(self, msg: CameraInfo) -> None:
        # self.get_logger().info("Got height {}".format(msg.height))
        return

    def wrenchCb(self, msg: Wrench) -> None:
        return

    def imgCb_(self, msg: Image) -> None:
        img = self.bridge_.imgmsg_to_cv2(msg, "bgr8")
        scale = .25
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        img = image_to_tensor(img, False).float()/255.
        try:
            center, radius = self.circle_(img, N=100, reduction=None)
            box = max_rectangle_in_circle(img.shape, center, radius)
            crp = crop_and_resize(img, box, [320, 320])
        except Exception as e:
            self.get_logger().warn(e)
            return

        if self.prev_crp_ is not None:
            G, duv = self.h_est_(crp, self.prev_crp_)
            blend = yt_alpha_blend(self.prev_crp_, warp_perspective(crp, G.inverse(), crp.shape[-2:]))
            blend = tensor_to_image(blend.cpu(), keepdim=False)
            cv2.imshow("blend", blend)

            # publish as numpy array
            G = G.squeeze().numpy()
            self.hom_pub_.publish(homographyToMsg(G))

            # class probability
            class_prob = Float64MultiArray(
                layout=MultiArrayLayout(
                    dim=[
                        MultiArrayDimension(label="class_probability", size=2)
                    ],
                    data_offset=0
                ),
                data=[1., 0.]
            )

            self.class_prob_pub_.publish(class_prob)
            
        self.prev_crp_ = crp

        img = tensor_to_image(img, False)
        # crp = tensor_to_image(crp, False)
        center, radius = center.int().cpu().numpy(), radius.int().cpu().numpy()

        cv2.circle(img, (center[0, 1], center[0, 0]), radius[0], (255, 255, 0), 2)

        cv2.imshow("img", img)
        # cv2.imshow("crp", crp)
        cv2.waitKey(1)
        

def main(args=None):
    rclpy.init(args=args)
    h_gen_node = HGenNode("h_gen_node")
    rclpy.spin(h_gen_node)
    h_gen_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
