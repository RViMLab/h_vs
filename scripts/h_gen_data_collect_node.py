#!/usr/bin/python3

import cv2
import numpy as np
import rclpy
import cv_bridge
import torch
from typing import List
import kornia
from kornia import image_to_tensor, tensor_to_image
from kornia.geometry import crop_and_resize, warp_perspective
from rclpy.node import Node
from std_msgs.msg import MultiArrayLayout, MultiArrayDimension, Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Wrench

from endoscopy import BoundingCircleDetector, HomographyEstimator
from endoscopy.utils import MODEL, max_rectangle_in_circle, yt_alpha_blend

from homography_generators import BaseHomographyGenerator
from homography_generators.utils.conversions import mat3DToMsg, updateCroppedPrincipalPoint, updateScaledPrincipalPoint
from homography_generators.utils.feature_homography import FeatureHomographyEstimation

import tkinter as tk


class HGenDataCollectNode(Node):
    inf_sub_: rclpy.subscription.Subscription
    img_sub_: rclpy.subscription.Subscription
    hom_pub_: rclpy.publisher.Publisher
    k_pub_: rclpy.publisher.Publisher
    wrench_sub_: rclpy.subscription.Subscription
    class_prob_pub_: rclpy.publisher.Publisher
    bridge_: cv_bridge.CvBridge
    circle_: BoundingCircleDetector
    center_buffer_: List[torch.Tensor]
    radius_buffer_: List[torch.Tensor]
    buffer_len_: int
    cam_info_: CameraInfo
    window_: tk.Tk

    def __init__(self, node_name: str="h_gen_node"):
        super().__init__(node_name=node_name)
        self.device_ = "cuda"
        self.inf_sub_ = self.create_subscription(CameraInfo, "~/camera_info", self.infCb_, rclpy.qos.qos_profile_system_default)  # launch in node namespace ~/*
        self.img_sub_ = self.create_subscription(Image, "~/image_raw", self.imgCb_, rclpy.qos.qos_profile_system_default)
        self.hom_pub_ = self.create_publisher(Float64MultiArray, "~/G", rclpy.qos.qos_profile_system_default)
        self.k_pub_ = self.create_publisher(Float64MultiArray, "~/K", rclpy.qos.qos_profile_system_default)
        self.wrench_sub_ = self.create_subscription(Wrench, "~/wrench", self.wrenchCb, rclpy.qos.qos_profile_system_default)
        self.class_prob_pub_ = self.create_publisher(Float64MultiArray, "~/class_probabilities", rclpy.qos.qos_profile_system_default)
        self.bridge_ = cv_bridge.CvBridge()
        self.circle_ = BoundingCircleDetector(model=MODEL.SEGMENTATION.UNET_RESNET_34, device=self.device_)

        self.window_ = tk.Tk()

        self.get_logger().info("loaded model")

        self.center_buffer_ = []
        self.radius_buffer_ = []
        self.buffer_len_ = 40

        self.cam_info_ = CameraInfo()

        self.init_ = False

        self.prev_crp_ = None

        # which model to use
        self.deep_hom_ = False
        self.orb_ = False
        self.transformer_ = True
        self.regi_ = False

        if self.deep_hom_:
            self.h_est_ = HomographyEstimator(model=MODEL.HOMOGRAPHY_ESTIMATION.H_48_RESNET_34, device=self.device_)
        if self.orb_:
            self.fd_ = cv2.ORB_create()
            self.fh_est_ = FeatureHomographyEstimation(self.fd_)
        if self.transformer_:
            # transformer loftr
            self.loftr_ = kornia.feature.LoFTR()
            self.ransac_ = kornia.geometry.RANSAC(inl_th=1.0)
            self.loftr_ = self.loftr_.to(self.device_)
            self.ransac_ = self.ransac_.to(self.device_)
        if self.regi_:
            self.registrator_ = kornia.geometry.ImageRegistrator('similarity', num_iterations=1)


        # create gui
        # self.window_.


        self.window_.mainloop()        

    def infCb_(self, msg: CameraInfo) -> None:
        self.cam_info_ = msg

    def wrenchCb(self, msg: Wrench) -> None:
        return

    def imgCb_(self, msg: Image) -> None:
        K = self.cam_info_.k.reshape([3, 3])
        d = np.array(self.cam_info_.d.tolist())
        shape = [self.cam_info_.height, self.cam_info_.width]

        img = self.bridge_.imgmsg_to_cv2(msg, "bgr8")

        # ideally: undistort image, and update camera matrix
        h, w = img.shape[:2]
        K, roi = cv2.getOptimalNewCameraMatrix(self.cam_info_.k.reshape([3, 3]), d, (w,h), 1, (w,h))  # alpha set to 1.

        img = cv2.undistort(img, self.cam_info_.k.reshape([3, 3]), d, None, K)

        scale = 1.
        resize_shape = [240, 320]
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

        img = image_to_tensor(img, False).float()/255.
        try:
            center, radius = self.circle_(img, N=100, reduction=None)
            if len(self.center_buffer_) >= self.buffer_len_:
                self.center_buffer_.pop(0)
                self.radius_buffer_.pop(0)

            # compute running average on center and radius
            self.center_buffer_.append(center)
            self.radius_buffer_.append(radius)
            center = torch.concat(self.center_buffer_).mean(axis=0).unsqueeze(0)
            radius = torch.concat(self.radius_buffer_).mean(axis=0).unsqueeze(0)
            box = max_rectangle_in_circle(img.shape, center, radius)  # shape: Bx4x2

            # update camera intrinsics in h_vs_node based on crop
            Kp = updateScaledPrincipalPoint(shape, resize_shape, updateCroppedPrincipalPoint(box[0, 0].cpu().numpy(), K))
            self.k_pub_.publish(mat3DToMsg(Kp))

            # publish updated camera matrix



            crp = crop_and_resize(img, box, resize_shape).to(self.device_)

            # to image
            if self.orb_:
                crp = (tensor_to_image(crp.cpu(), keepdim=False)*255.).astype(np.uint8)
        except Exception as e:
            self.get_logger().warn(e)
            return

        if self.prev_crp_ is not None:
            # deep homography estimator
            if self.deep_hom_:
                G, duv = self.h_est_(self.prev_crp_, crp)
                blend = yt_alpha_blend(self.prev_crp_, warp_perspective(crp, G, crp.shape[-2:]))
                blend = tensor_to_image(blend.cpu(), keepdim=False)
                G = G.squeeze().numpy()
                self.hom_pub_.publish(mat3DToMsg(np.linalg.inv(G)))  # inverse!!
            
            # opencv orb
            if self.orb_:
                G, duv = self.fh_est_ (self.prev_crp_, crp)
                blend = yt_alpha_blend(self.prev_crp_.astype(float)/255., cv2.warpPerspective(crp.astype(float)/255., np.linalg.inv(G), (crp.shape[1], crp.shape[0])))
                self.hom_pub_.publish(mat3DToMsg(G))
            
            # tranformer model
            if self.transformer_:
                with torch.no_grad():
                    input = {"image0": kornia.color.rgb_to_grayscale(self.prev_crp_), "image1": kornia.color.rgb_to_grayscale(crp)}
                    correspondence_dict = self.loftr_(input)
                    G, mask = self.ransac_(correspondence_dict["keypoints0"], correspondence_dict["keypoints1"])
                    blend = yt_alpha_blend(self.prev_crp_, warp_perspective(crp, G.inverse().unsqueeze(0), crp.shape[-2:]))
                    blend = tensor_to_image(blend.cpu(), keepdim=False)
                    G = G.squeeze().cpu().numpy()
                    self.hom_pub_.publish(mat3DToMsg(G)) 
            if self.regi_:
                G = self.registrator_.register(self.prev_crp_, crp)
                blend = yt_alpha_blend(self.prev_crp_, warp_perspective(crp, G, crp.shape[-2:]))
                blend = tensor_to_image(blend.cpu(), keepdim=False)
                G = G.squeeze().detach().cpu().numpy()
                self.hom_pub_.publish(mat3DToMsg(G)) 
  
            cv2.imshow("blend", blend)

            # class probability
            class_prob = Float64MultiArray(
                layout=MultiArrayLayout(
                    dim=[
                        MultiArrayDimension(label="class_probabilities", size=4)
                    ],
                    data_offset=0
                ),
                data=[0., 0., 1., 1.]
            )

            self.class_prob_pub_.publish(class_prob)



        # set weights/cases
        # run visual servo, based on target image
        # record
        
        
        if not self.init_:
            self.init_ = True
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
    h_gen_node = HGenDataCollectNode("h_gen_data_collect_node")  # blocked by tkinter mainloop
    # rclpy.spin(h_gen_node)
    h_gen_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
