#!/usr/bin/python3

import rospy
import cv2
import numpy as np
import camera_info_manager
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import homography_generators.calibration_pattern_homography_generator as cphg
from homography_generators.endoscopy import endoscopy


class ImageHandler():
    def __init__(self):
        self._img0 = np.array([])
        self._img = np.array([])

        self._img0_init = False

        self.cv_bridge = CvBridge()

        self._img0_sub = rospy.Subscriber('visual_servo/img0', Image, self._img0_cb)
        self._img_sub = rospy.Subscriber('camera/image_raw', Image, self._img_cb)

    def _img0_cb(self, msg):
        if not self._img0_init: 
            self._img0 = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self._img0_init = True


    def _img_cb(self, msg):
        self._img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    @property
    def Img0(self):
        return self._img0

    @property
    def Img(self):
        return self._img


if __name__ == '__main__':

    rospy.init_node('h_gen_endoscopy_calibration_pattern_node')

    cname = rospy.get_param("h_gen_endoscopy_calibration_pattern_node/cname")
    url = rospy.get_param("h_gen_endoscopy_calibration_pattern_node/url")

    camera_info_manager = camera_info_manager.CameraInfoManager(cname, url)
    camera_info_manager.loadCameraInfo()  # explicitely load info
    camera_info = camera_info_manager.getCameraInfo()

    K = np.asarray(camera_info.K).reshape([3,3])
    D = np.asarray(camera_info.D)

    # Initialize homography generator
    hg = cphg.CalibrationPatternHomographyGenerator(K=K, D=D, undistort=False)  # undistort manually below

    # Handle initial and current images
    shape = [camera_info.height, camera_info.width, 3]

    ih = ImageHandler()

    # Crop endoscopic view
    tracker = endoscopy.CoMBoundaryTracker()

    # Wait for initialization
    initialized = False
    while not initialized:
        if ih.Img0.shape[0] == 0:
            rospy.sleep(rospy.Duration(0.1))
            continue

        img0, _ = hg.undistort(ih.Img0)
        mask = endoscopy.bilateralSegmentation(img0.astype(np.uint8), th=0.1)
        center, radius = tracker.updateBoundaryCircle(mask)

        if radius is None:
            rospy.loginfo('h_gen_endoscopy_calibration_pattern_node: Endoscopic view not initialized.')
            continue

        inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(mask.shape, center, radius)
        inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

        img0 = endoscopy.crop(img0, inner_top_left, inner_shape)
        img0 = cv2.resize(img0, (640, 480))

        initialized = True

    # Publish desired projective homography and visual error
    homography_pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)
    error_pub = rospy.Publisher("visual_servo/mean_pairwise_distance", Float64, queue_size=1)

    while not rospy.is_shutdown():
        # Show initial and desired images
        cv2.namedWindow('Initial Image')
        cv2.namedWindow('Current Undistorted Image')
        cv2.namedWindow('Error Image')

        # Update with current image and compute desired projective homography
        img, _ = hg.undistort(ih.Img)
        mask = endoscopy.bilateralSegmentation(img.astype(np.uint8), th=0.1)
        center, radius = tracker.updateBoundaryCircle(mask)

        inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(mask.shape, center, radius)
        inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

        img = endoscopy.crop(img, inner_top_left, inner_shape)
        img = cv2.resize(img, (640, 480))

        hg.addImg(img)
        G, mean_pairwise_distance = hg.desiredHomography(img0)

        cv2.imshow('Initial Image', img0)
        cv2.imshow('Current Undistorted Image', hg.Imgs[0])  # undistorted
        cv2.imshow('Error Image', cv2.warpPerspective(img0, G, (hg.Imgs[0].shape[1], hg.Imgs[0].shape[0])) - hg.Imgs[0])
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
