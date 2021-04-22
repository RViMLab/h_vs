#!/usr/bin/python3

import rospy
import cv2
# import torch
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import camera_info_manager

import homography_generators.calibration_pattern_homography_generator as cphg

# write node that 
#   subscribes camera images, possibly holds
#   read cam params
#   undistort img
#   computes projective homography

# TODO: https://gitter.im/RoboStack/Lobby, try to add a conda env once more ros packages integrated

# img = np.array([])
# img0 = np.array([])


class ImageHandler():
    def __init__(self, img0, img):
        self._img0 = img0
        self._img = img

        self._cv_bridge = CvBridge()

        self._img0_sub = rospy.Subscriber('visual_servo/img0', Image, self._img0_cb)
        self._img_sub = rospy.Subscriber('endoscope_camera/image_raw', Image, self._img_cb)

    def _img0_cb(self, msg):
        self._img0 = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    def _img_cb(self, msg):
        self._img = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    @property
    def Img0(self):
        return self._img0

    @property
    def Img(self):
        return self._img


if __name__ == '__main__':

    rospy.init_node('h_gen_node')
    
    cname = rospy.get_param("h_gen_node/cname")
    url = rospy.get_param("h_gen_node/url")

    camera_info_manager = camera_info_manager.CameraInfoManager(cname, url)
    camera_info_manager.loadCameraInfo()  # explicitely load info
    camera_info = camera_info_manager.getCameraInfo()

    K = np.asarray(camera_info.K).reshape([3,3])
    D = np.asarray(camera_info.D)

    # Initialize homography generator with intrinsics
    hg = cphg.CalibrationPatternHomographyGenerator(K=K, D=D)

    shape = [camera_info.height, camera_info.width, 3]
    img0 = np.zeros(shape)
    img = np.zeros(shape)

    ih = ImageHandler(img0, img)

    pub = rospy.Publisher("visual_servo/G", numpy_msg(Float64MultiArray), queue_size=1)

    while not rospy.is_shutdown():
        cv2.imshow('img0', ih.Img0)
        cv2.imshow('img', ih.Img)
        cv2.waitKey(1)

        hg.addImg(ih.Img)
        G = hg.desiredHomography(ih.Img0)

        layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='rows', size=G.shape[0]),
                MultiArrayDimension(label='cols', size=G.shape[1])
            ],
            data_offset=0
        )

        msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())

        pub.publish(msg)
