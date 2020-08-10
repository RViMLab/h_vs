#!/usr/bin/python3

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import homography_generators.calibration_pattern_homography_generator as cphg


class ImageHandler():
    def __init__(self, img0, img):
        self.img0 = img0
        self.img = img

        self.cv_bridge = CvBridge()

        self.img0_sub = rospy.Subscriber('visual_servo/img0', Image, self.img0_cb)
        self.img_sub = rospy.Subscriber('camera/image_raw', Image, self.img_cb)

    def img0_cb(self, msg):
        self.img0 = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    def img_cb(self, msg):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")


if __name__ == '__main__':

    rospy.init_node('h_gen_calibration_pattern_node')

    # Get camera intrinsics
    K = np.array(rospy.get_param('camera_matrix/data')).reshape([
        rospy.get_param('camera_matrix/rows'),
        rospy.get_param('camera_matrix/cols')
    ])

    d = np.array(rospy.get_param('distortion_coefficients/data'))
     
    # Initialize homography generator with intrinsics
    hg = cphg.CalibrationPatternHomographyGenerator(K=K, d=d)

    # Handle initial and current images
    shape = [rospy.get_param('image_height'), rospy.get_param('image_width'), 3]
    img0 = np.zeros(shape)
    img = np.zeros(shape)

    ih = ImageHandler(img0, img)

    # Publish desired projective homography
    pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)

    while not rospy.is_shutdown():
        # Show initial and desired images
        cv2.imshow('img0', ih.img0)
        cv2.imshow('img', ih.img)
        cv2.waitKey(1)

        # Update with current image and compute desired projective homography
        hg.addImg(ih.img)
        G = hg.desiredHomography(ih.img0)

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
