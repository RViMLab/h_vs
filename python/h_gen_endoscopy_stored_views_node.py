#!/usr/bin/python3

import os
import cv2
import numpy as np
from typing import Tuple
import networkx as nx
import pandas as pd
import rospy
import actionlib
import camera_info_manager
from rospy.topics import Subscriber
from std_msgs.msg import Int32, Float64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge

from homography_generators.base_homography_generator import BaseHomographyGenerator
import homography_generators.stored_view_homography_generator as svhg
from homography_generators.endoscopy import endoscopy
from homography_generators.homography_imitation_learning.utils import yt_alpha_blend
from h_vs.srv import k_intrinsics, k_intrinsicsRequest, capture, captureRequest, captureResponse
from h_vs.msg import h_vsAction, h_vsGoal, h_vsFeedback, h_vsResult, pairwise_distance
from rcm_msgs.msg import rcm


class StoredViewsActionServer(object):
    def __init__(self,
        hg: BaseHomographyGenerator,
        mpd_th: int=5,
        mpd_th_final: int=1.5,
        resize_shape: tuple=(240, 320),
        pre_process: bool=True,
        h_rcm_vs_state: str='h_rcm_vs/RCM_ActionServer/state',
        img_topic: str='camera/image_raw',
        g_topic: str='visual_servo/G',
        e_topic: str='visual_servo/pairwise_distance',
        b_topic: str='visual_servo/blend',
        j_topic: str='joint_states',
        intrinsic_service: str='visual_servo/K',
        cap_service: str='visual_servo/capture',
        action_server: str='visual_servo/execute',
        log_path: str='/tmp/h_gen_endoscopy_stored_views'
    ):
        # homography generator
        self._hg = hg

        # convergence threshold
        self._mpd_th = mpd_th
        self._mpd_th_final = mpd_th_final

        # resize shape and pre-processing
        self._resize_shape = resize_shape
        self._pre_process = pre_process

        # rcm state subscriber
        self._rcm_state_sub = Subscriber(h_rcm_vs_state, rcm, self._rcm_state_cb)
        self._rcm_state = rcm()
        self._rcm_state_dict = {}
        self._log_df = pd.DataFrame(columns=['id', 'path', 'target_img', 'final_img', 'target_rcm', 'final_rcm', 'target_joint_state', 'final_joint_state'])
        self._log_idx = 0

        # image stream handler
        self._img = np.array([])
        self._cv_bridge = CvBridge()
        self._img_topic = img_topic
        self._img_sub = rospy.Subscriber(self._img_topic, Image, self._img_cb)

        # publish desired projective homography and visual error
        self._g_topic = g_topic
        self._homography_pub = rospy.Publisher(self._g_topic, Float64MultiArray, queue_size=1)
        self._e_ropic = e_topic
        self._error_pub = rospy.Publisher(self._e_ropic, pairwise_distance, queue_size=1)

        # image blend topic
        self._b_topic = b_topic
        self._blend_pub = rospy.Publisher(self._b_topic, Image, queue_size=1)

        # joint state topic
        self._joint_state = JointState()
        self._joint_state_dict = {}
        self._j_topic = j_topic
        self._joint_state_sub = rospy.Subscriber(self._j_topic, JointState, self._joint_state_cb)

        # create service proxy to update camera intrinsics in h_vs
        self._intrinsic_service = intrinsic_service
        rospy.loginfo('h_gen_endoscopy_stored_views_node: Waiting for K service server...')
        rospy.wait_for_service(self._intrinsic_service)
        rospy.loginfo('h_gen_endoscopy_stored_views_node: Done.')
        self._intrinsic_client = rospy.ServiceProxy(self._intrinsic_service, k_intrinsics)

        # crop endoscopic view
        self._tracker = endoscopy.CoMBoundaryTracker()

        # image capture service (extends graph)
        self._cap_service = cap_service
        self._cap_serv = rospy.Service(self._cap_service, capture, self._cap_cb)

        # action server, see http://wiki.ros.org/actionlib_tutorials/Tutorials/Writing%20a%20Simple%20Action%20Server%20using%20the%20Execute%20Callback%20%28Python%29
        self._action_server = action_server
        self._as = actionlib.SimpleActionServer(self._action_server, h_vsAction, execute_cb=self._execute_cb, auto_start=False)
        self._as.start()

        self._log_path = log_path
        if not os.path.exists(self._log_path):
            os.mkdir(self._log_path)

    def _rcm_state_cb(self, msg: rcm):  # keep updated state
        self._rcm_state = msg

    def _process_endoscopic_image(self, img: np.ndarray, resize_shape: tuple=(480, 640)) -> Tuple[np.ndarray, np.ndarray]:
        r"""Undistorts an endoscopic view, crops, and updates the camera matrix.

        Args:
            img (np.ndarray): Image to be processed
            resize_shape (tuple): Desired image shape

        Return:
            img, K_pp (Tuple[np.ndarray, np.ndarray]): Cropped, undistorted image, updated camera matrix
        """

        # Update with current image and compute desired projective homography
        img, K_p = self._hg.undistort(img)
        if not self._tracker.initCircle:
            mask = endoscopy.bilateralSegmentation(img.astype(np.uint8), th=0.1)
            center, radius = self._tracker.updateBoundaryCircle(mask)

            if radius is None:
                return np.array([]), np.array([])
            

        # # if radius is None:
        # #     return np.array([]), np.array([])
        # else:
        center, radius = self._tracker.circle

        inner_top_left, inner_shape = endoscopy.maxRectangleInCircle(img.shape, center, radius)
        inner_top_left, inner_shape = inner_top_left.astype(np.int), tuple(map(np.int, inner_shape))

        img = endoscopy.crop(img, inner_top_left, inner_shape)

        K_pp = endoscopy.updateCroppedPrincipalPoint(inner_top_left, K_p)  # update camera intrinsics under cropping
        K_pp = endoscopy.updateScaledPrincipalPoint(img.shape, resize_shape, K_p)  # update camera intrinsics under scaling
        img = cv2.resize(img, (resize_shape[1], resize_shape[0]))

        return img, K_pp

    def _build_multiarray(self, mat: np.ndarray) -> Float64MultiArray:
        r"""Build multi array from numpy array.

        Args:
            mat (np.ndarray): Matrix to be transformed

        Return:
            msg (Float64MultiArray): Message containing mat 
        """
        layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label='rows', size=mat.shape[0]),
                MultiArrayDimension(label='cols', size=mat.shape[1])
            ],
            data_offset=0
        )
        msg = Float64MultiArray(layout=layout, data=mat.flatten().tolist())
        return msg   

    def _build_intrinsic_message(self, K: np.ndarray) -> k_intrinsicsRequest:
        r"""Builds request message to update camera intrinsics.

        Args:
            K (np.ndarray): Camera intrinsics
        
        Return:
            req (k_intrinsicRequest): Request message contraining camera intrinsics
        """
        # update camera intrinsics via service call
        msg = self._build_multiarray(K)
        req = k_intrinsicsRequest()
        req.K = msg
        return req

    def _img_cb(self, msg: Image) -> None:
        r"""Keeps the current image as numpy array.
        """
        img = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        if self._pre_process:
            img, K_pp = self._process_endoscopic_image(img, resize_shape=self._resize_shape)
            if img.shape[0] is not 0:
                K_pp_req = self._build_intrinsic_message(K_pp)
                self._intrinsic_client(K_pp_req)  # update camera intrinsics in h_vs
                self._img = img
        else:
            self._img = img

    def _joint_state_cb(self, msg: JointState) -> None:
        r"""Keep copy of current joint state.
        """
        self._joint_state = msg

    def _cap_cb(self, req: captureRequest) -> captureResponse:
        r"""Capture callback. Add current image to graph on capture call.
        """
        wrp = self._img
        if wrp.shape[0] is 0:
            rospy.loginfo('{}: Capture request failed. Endoscopic view not initialized yet, check lighting.'.format(self._action_server))
            res = captureResponse()
            res.success.data = False
            return res

        wrp = self._cv_bridge.cv2_to_imgmsg(wrp)

        # add image to graph, get current id and respond to request
        id = self._hg.addImg(wrp)
        res = captureResponse()
        res.capture = wrp
        res.id = Int32(id)
        res.success.data = True

        # add rcm state to state dict
        self._rcm_state_dict[id] = self._rcm_state

        # add joint state to state dict
        self._joint_state_dict[id] = self._joint_state

        return res

    def _execute_cb(self, goal: h_vsGoal) -> None:
        # read goal id and find path from current node
        src_id = self._hg.ID
        target_id = goal.id.data

        try:
            path = nx.dijkstra_path(self._hg.ImgGraph, src_id, target_id)
        except:
            rospy.loginfo('{}: Failed to find path.'.format(self._action_server))
            self._as.set_aborted()
            return

        rospy.loginfo('{}: Found path from {} to {}: {}'.format(self._action_server, src_id, target_id, path))

        # execution loop
        reached = False
        checkpoint = 0
        while not reached:
            if self._as.is_preempt_requested():
                rospy.loginfo('{}: Preempted.'.format(self._action_server))
                self._as.set_preempted()
                return

            # poll current view
            wrp = self._img

            # compute visual servo
            G, duv, mean_pairwise_distance, std_pairwise_distance, n_matches = self._hg.desiredHomography(wrp, id=path[checkpoint])

            if mean_pairwise_distance is not None:
                # publish blend
                target = self._cv_bridge.imgmsg_to_cv2(self._hg.ImgGraph.nodes[path[checkpoint]]['data'])
                target_est = cv2.warpPerspective(wrp, np.linalg.inv(G), (wrp.shape[1], wrp.shape[0]))
                blend = yt_alpha_blend(target/255., target_est/255.)
                self._blend_pub.publish(self._cv_bridge.cv2_to_imgmsg((blend*255).astype(np.uint8), 'bgr8'))

                self._error_pub.publish(pairwise_distance(Float64(mean_pairwise_distance), Float64(std_pairwise_distance), Int32(n_matches)))

                # publish feedback
                feedback = h_vsFeedback()
                feedback.id.data = self._hg.ID
                feedback.mpd.data = mean_pairwise_distance
                feedback.path.data = path
                self._as.publish_feedback(feedback)


                if checkpoint >= len(path) - 1:
                    mpd = self._mpd_th_final
                else:
                    mpd = self._mpd_th

                if mean_pairwise_distance < mpd or rospy.is_shutdown():
                    self._hg.ID = path[checkpoint]  # update current node
                    rospy.loginfo('{}: Checkpoint reached. New node: {}. Current mean pairwise distance: {:.1f}'.format(self._action_server, self._hg.ID, mean_pairwise_distance))
                    checkpoint += 1  # update next checkpoint
                    
                    # log target and current view, also log current+target pose and joint state
                    target = self._cv_bridge.imgmsg_to_cv2(self._hg.ImgGraph.nodes[self._hg.ID]['data'])

                    self._log_df = self._log_df.append({
                        'id': self._hg.ID,
                        'time': rospy.Time.now().to_sec(),
                        'path': path,
                        'target_img': target,
                        'final_img': wrp,
                        'target_rcm': self._rcm_state_dict[self._hg.ID],
                        'final_rcm': self._rcm_state,
                        'target_joint_state': self._joint_state_dict[self._hg.ID],
                        'final_joint_state': self._joint_state
                    }, ignore_index=True)

                    if checkpoint >= len(path) or rospy.is_shutdown():
                        rospy.loginfo('{}: Desired view reached, final mean pairwise distance: {:.1f}'.format(self._action_server, mean_pairwise_distance))
                        reached = True

                        # publish steady state
                        msg = self._build_multiarray(np.eye(3))
                        self._homography_pub.publish(msg)

                        # save results and clear df
                        self._log_df.to_pickle('{}/path_{}.pkl'.format(self._log_path, self._log_idx), protocol=2)
                        self._log_df.drop(self._log_df.index, inplace=True)
                        self._log_idx += 1

                else:  # execute motion
                    msg = self._build_multiarray(G)
                    self._homography_pub.publish(msg)
            else:
                rospy.loginfo('{}: No homography found.'.format(self._action_server))
                rospy.sleep(rospy.Duration(0.01))

        result = h_vsResult()
        result.id.data = self._hg.ID
        result.mpd.data = mean_pairwise_distance
        result.path.data = path
        self._as.set_succeeded(result)


if __name__ == '__main__':

    rospy.init_node('h_gen_endoscopy_stored_views_node')

    cname = rospy.get_param("h_gen_endoscopy_stored_views_node/cname")
    url = rospy.get_param("h_gen_endoscopy_stored_views_node/url")
    sim = rospy.get_param("h_gen_endoscopy_stored_views_node/sim")

    camera_info_manager = camera_info_manager.CameraInfoManager(cname, url)
    camera_info_manager.loadCameraInfo()  # explicitely load info
    camera_info = camera_info_manager.getCameraInfo()

    K = np.asarray(camera_info.K).reshape([3,3])
    D = np.asarray(camera_info.D)

    # Initialize homography generator
    hg = svhg.StoredViewHomographyGenerator(K=K, D=D, undistort=False)  # undistort manually below

    # Start action server
    action_server = StoredViewsActionServer(hg, pre_process=(not sim))

    rospy.spin()
