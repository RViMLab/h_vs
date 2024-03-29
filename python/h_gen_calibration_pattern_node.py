#!/usr/bin/python3

import rospy
import cv2
import numpy as np
import camera_info_manager
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import homography_generators.calibration_pattern_homography_generator as cphg


class ImageHandler():
    def __init__(self, img0, img):
        self._img0 = img0
        self._img = img

        self._cv_bridge = CvBridge()

        self._img0_sub = rospy.Subscriber('visual_servo/img0', Image, self._img0_cb)
        self._img_sub = rospy.Subscriber('camera/image_raw', Image, self._img_cb)


    def _img0_cb(self, msg):
        self._img0 = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def _img_cb(self, msg):
        self._img = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    @property
    def Img0(self):
        return self._img0

    @property
    def Img(self):
        return self._img


if __name__ == '__main__':

    rospy.init_node('h_gen_calibration_pattern_node')

    cname = rospy.get_param("h_gen_calibration_pattern_node/cname")
    url = rospy.get_param("h_gen_calibration_pattern_node/url")

    camera_info_manager = camera_info_manager.CameraInfoManager(cname, url)
    camera_info_manager.loadCameraInfo()  # explicitely load info
    camera_info = camera_info_manager.getCameraInfo()

    K = np.asarray(camera_info.K).reshape([3,3])
    D = np.asarray(camera_info.D)

    # Initialize homography generator
    hg = cphg.CalibrationPatternHomographyGenerator(K=K, D=D, undistort=True)

    # Handle initial and current images
    shape = [camera_info.height, camera_info.width, 3]
    img0 = np.zeros(shape)
    img = np.zeros(shape)

    ih = ImageHandler(img0, img)

    # Publish desired projective homography and visual error
    homography_pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)
    error_pub = rospy.Publisher("visual_servo/mean_pairwise_distance", Float64, queue_size=1)

    while not rospy.is_shutdown():
        # Show initial and desired images
        cv2.namedWindow('Initial Image')
        cv2.namedWindow('Current Undistorted Image')
        cv2.namedWindow('Error Image')

        # Update with current image and compute desired projective homography
        hg.addImg(ih.Img)
        G, mean_pairwise_distance = hg.desiredHomography(ih.Img0)

        cv2.imshow('Initial Image', ih.Img0)
        cv2.imshow('Current Undistorted Image', hg.ImgGraph.nodes[0]['data'])  # undistorted
        cv2.imshow('Error Image', ih.Img0 - ih.Img)
        # cv2.imshow('Error Image', cv2.warpPerspective(ih.Img0, G, (ih.Img.shape[1], ih.Img.shape[0])) - ih.Img)
        cv2.waitKey(1)

        # Publish projective homography
        layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='rows', size=G.shape[0]),
                MultiArrayDimension(label='cols', size=G.shape[1])
            ],
            data_offset=0
        )
        msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())
        homography_pub.publish(msg)
        if mean_pairwise_distance:
            error_pub.publish(mean_pairwise_distance)
