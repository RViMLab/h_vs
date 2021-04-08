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
    def __init__(self, img0, img):
        self._img0 = img0
        self._img = img

        self.cv_bridge = CvBridge()

        self._img0_sub = rospy.Subscriber('visual_servo/img0', Image, self._img0_cb)
        self._img_sub = rospy.Subscriber('camera/image_raw', Image, self._img_cb)

    def _img0_cb(self, msg):
        self._img0 = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        mask = endoscopy.bilateralSegmentation(self._img0, 0.2)
        center, radius = endoscopy.boundaryCircle(mask, 10)
        top_left, shape = endoscopy.maxRectanlgeInCircle(mask.shape, center, radius)
        self._img0 = endoscopy.crop(self._img0, top_left, shape)

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
    img0 = np.zeros(shape)
    img = np.zeros(shape)

    ih = ImageHandler(img0, img)

    # Crop endoscopic view
    tracker = endoscopy.CoMBoundaryTracker()

    # Publish desired projective homography and visual error
    homography_pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)
    error_pub = rospy.Publisher("visual_servo/mean_pairwise_distance", Float64, queue_size=1)

    while not rospy.is_shutdown():
        # Show initial and desired images
        cv2.namedWindow('Initial Image')
        cv2.namedWindow('Current Undistorted Image')
        cv2.namedWindow('Error Image')

        # Update with current image and compute desired projective homography
        img = hg.undistort(ih.Img)
        mask = endoscopy.bilateralSegmentation(img.astype(np.uint8), th=0.1)
        center, radius = tracker.updateBoundaryCircle(mask)

        if radius is None:
            continue

        inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(mask.shape, center, radius)
        inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

        img = endoscopy.crop(img, inner_top_left, inner_shape)
        img = cv2.resize(img, (640, 480))
        
        hg.addImg(img)
        # G, mean_pairwise_distance = hg.desiredHomography(ih.Img0)

        cv2.imshow('Initial Image', ih.Img0)
        cv2.imshow('Current Undistorted Image', hg.Imgs[0])  # undistorted
        cv2.imshow('Error Image', ih.Img0 - ih.Img)
        # cv2.imshow('Error Image', cv2.warpPerspective(ih.Img0, G, (ih.Img.shape[1], ih.Img.shape[0])) - ih.Img)
        cv2.waitKey(1)

        # # Publish projective homography
        # layout = MultiArrayLayout(
        #     dim=[
        #         MultiArrayDimension(label='rows', size=G.shape[0]),
        #         MultiArrayDimension(label='cols', size=G.shape[1])
        #     ],
        #     data_offset=0
        # )
        # msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())
        # homography_pub.publish(msg)
        # if mean_pairwise_distance:
        #     error_pub.publish(mean_pairwise_distance)
