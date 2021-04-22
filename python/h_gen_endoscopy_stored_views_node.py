#!/usr/bin/python3

import rospy
import cv2
import numpy as np
import camera_info_manager
from std_srvs.srv import Empty
from std_msgs.msg import Int32, Float64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import homography_generators.calibration_pattern_homography_generator as cphg
from homography_generators.endoscopy import endoscopy
from h_vs.srv import k_intrinsics, k_intrinsicsRequest, capture, captureResponse


class ImageHandler():
    def __init__(self):
        self._img = np.array([])
        self._cv_bridge = CvBridge()
        self._img_sub = rospy.Subscriber('camera/image_raw', Image, self._img_cb)
        self._cap_srv = rospy.Service('visual_servo/capture', capture, self._cap_cb)

    def _img_cb(self, msg):
        self._img = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")

    def _cap_cb(self, req):
        res = captureResponse()
        res.capture = self._cv_bridge.cv2_to_imgmsg(self._img)
        res.id = Int32(0)
        return res

    @property
    def Img(self):
        return self._img


def switch_h_gen_cb(req):
    global gen_h
    gen_h = not gen_h
    return []


if __name__ == '__main__':

    rospy.init_node('h_gen_endoscopy_stored_views_node')

    cname = rospy.get_param("h_gen_endoscopy_stored_views_node/cname")
    url = rospy.get_param("h_gen_endoscopy_stored_views_node/url")

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

    # Disable homography generation
    gen_h = False
    switch_h_gen_server = rospy.Service('visual_servo/switch_h_gen', Empty, switch_h_gen_cb)

    # # Wait for initialization
    # initialized = False
    # while not initialized:
    #     if ih.Img0.shape[0] == 0:
    #         rospy.sleep(rospy.Duration(0.1))
    #         continue

    #     img0, K_p = hg.undistort(ih.Img0)
    #     mask = endoscopy.bilateralSegmentation(img0.astype(np.uint8), th=0.1)
    #     center, radius = tracker.updateBoundaryCircle(mask)

    #     if radius is None:
    #         rospy.loginfo('h_gen_endoscopy_stored_views_node: Endoscopic view not initialized.')
    #         continue

    #     inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(mask.shape, center, radius)
    #     inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

    #     img0 = endoscopy.crop(img0, inner_top_left, inner_shape)
    #     img0 = cv2.resize(img0, (640, 480))

    #     initialized = True

    # # Publish desired projective homography and visual error
    # homography_pub = rospy.Publisher("visual_servo/G", Float64MultiArray, queue_size=1)
    # error_pub = rospy.Publisher("visual_servo/mean_pairwise_distance", Float64, queue_size=1)

    # # Create service proxy to update camera intrinsics in h_vs
    # k_server = "visual_servo/K"
    # rospy.loginfo('h_gen_endoscopy_stored_views_node: Waiting for K service server...')
    # rospy.wait_for_service(k_server)
    # rospy.loginfo('h_gen_endoscopy_stored_views_node: Done.')
    # k_client = rospy.ServiceProxy(k_server, k_intrinsics)

    while not rospy.is_shutdown():
        if not gen_h:
            rospy.sleep(rospy.Duration(0.1))
            continue

    #     # Show initial and desired images
    #     cv2.namedWindow('Initial Image')
    #     cv2.namedWindow('Current Undistorted Image')
    #     cv2.namedWindow('Error Image')

    #     # Update with current image and compute desired projective homography
    #     img, K_p = hg.undistort(ih.Img)
    #     mask = endoscopy.bilateralSegmentation(img.astype(np.uint8), th=0.1)
    #     center, radius = tracker.updateBoundaryCircle(mask)

    #     inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(mask.shape, center, radius)
    #     inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

    #     img = endoscopy.crop(img, inner_top_left, inner_shape)

    #     resize_shape = (480, 640)
    #     K_pp = endoscopy.updateCroppedPrincipalPoint(inner_top_left, K_p)  # update camera intrinsics under cropping
    #     K_pp = endoscopy.updateScaledPrincipalPoint(img.shape, resize_shape, K_p)  # update camera intrinsics under scaling
    #     img = cv2.resize(img, (resize_shape[1], resize_shape[0]))

    #     hg.addImg(img)
    #     G, mean_pairwise_distance = hg.desiredHomography(img0)
    #     # wrp = cv2.warpPerspective(img0, G, (hg.ImgGraph.nodes[0]['data'].shape[1], hg.ImgGraph.nodes[0]['data'].shape[0]))

    #     cv2.imshow('Initial Image', img0)
    #     cv2.imshow('Current Undistorted Image', hg.ImgGraph.nodes[0]['data'])  # undistorted
    #     cv2.imshow('Error Image', (
    #         cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(hg.ImgGraph.nodes[0]['data'], cv2.COLOR_BGR2GRAY)
    #     ))
    #     cv2.waitKey(1)

    #     # Update camera intrinsics via service call
    #     layout = MultiArrayLayout(
    #         dim=[
    #             MultiArrayDimension(label='rows', size=K_pp.shape[0]),
    #             MultiArrayDimension(label='cols', size=K_pp.shape[1])
    #         ],
    #         data_offset=0
    #     )
    #     msg = Float64MultiArray(layout=layout, data=K_pp.flatten().tolist())

    #     req = k_intrinsicsRequest()
    #     req.K = msg

    #     res = k_client(req)

    #     # Publish projective homography
    #     layout = MultiArrayLayout(
    #         dim=[
    #             MultiArrayDimension(label='rows', size=G.shape[0]),
    #             MultiArrayDimension(label='cols', size=G.shape[1])
    #         ],
    #         data_offset=0
    #     )
    #     msg = Float64MultiArray(layout=layout, data=G.flatten().tolist())
    #     homography_pub.publish(msg)
    #     if mean_pairwise_distance:
    #         error_pub.publish(mean_pairwise_distance)
