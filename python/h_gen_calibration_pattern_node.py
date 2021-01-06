#!/usr/bin/python3

import rospy
import cv2
import numpy as np
import camera_info_manager
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import homography_generators.calibration_pattern_homography_generator as cphg


class ImageHandler():
    def __init__(self, img0, img):
        self._img0 = img0
        self._img = img

        self.cv_bridge = CvBridge()

        self._img0_sub = rospy.Subscriber('visual_servo/img0', Image, self._img0_cb)
        self._img_sub = rospy.Subscriber('camera/image_raw', Image, self._img_cb)


    def _img0_cb(self, msg):
        self._img0 = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    def _img_cb(self, msg):
        self._img = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

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

    camera_info = camera_info_manager.CameraInfoManager(cname, url)
    K = camera_info.getCameraInfo().K.reshape([3,3])
    D = camera_info.getCameraInfo().D

    # Initialize homography generator
    hg = cphg.CalibrationPatternHomographyGenerator(K=K, D=D, undistort=True)

    # Handle initial and current images
    shape = [rospy.get_param('image_height'), rospy.get_param('image_width'), 3]
    img0 = np.zeros(shape)
    img = np.zeros(shape)

    ih = ImageHandler(img0, img)

    # Publish desired projective homography
    pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)

    while not rospy.is_shutdown():
        # Show initial and desired images
        cv2.namedWindow('Initial Image')
        cv2.namedWindow('Current Image')
        cv2.namedWindow('Error Image')

        cv2.moveWindow('Initial Image', 560, 100)
        cv2.moveWindow('Current Image', 950, 100)
        cv2.moveWindow('Error Image', 1335, 100)

        cv2.imshow('Initial Image', ih.img0)
        cv2.imshow('Current Image', ih.img)
        cv2.imshow('Error Image', ih.img0 - ih.img)
        cv2.waitKey(1)

        # Update with current image and compute desired projective homography
        hg.addImg(ih.Img)
        G = hg.desiredHomography(ih.Img0)

        # Publish projective homography
        layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='rows', size=G.shape[0]),
                MultiArrayDimension(label='cols', size=G.shape[1])
            ],
            data_offset=0
        )
        msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())
        pub.publish(msg)
