import numpy as np
import cv2
from typing import Tuple
from cv_bridge import CvBridge

from homography_generators.base_homography_generator import BaseHomographyGenerator
from homography_generators.homography_imitation_learning import utils

# from homography_generators.homography_imitation_learning.utils import yt_alpha_blend


class StoredViewHomographyGenerator(BaseHomographyGenerator):
    def __init__(self, K: np.ndarray, D: np.ndarray, undistort: bool=False) -> None:
        super().__init__(K, D, buffer_size=None, undistort=undistort)

        # self._feature_detector = cv2.xfeatures2d.SIFT_create()  # proprietary, how to use?
        self._feature_detector = cv2.xfeatures2d.SURF_create()  # proprietary, how to use?
        # self._feature_detector = cv2.ORB_create(nfeatures=2000)
        # self._feature_detector = cv2.FastFeatureDetector_create()

        self._cv_bridge = CvBridge()
        self._feature_homography = utils.FeatureHomographyEstimation(self._feature_detector)

    def desiredHomography(self, wrp: np.ndarray, id: int) -> Tuple[np.ndarray, np.ndarray, np.float64, np.float64, np.int32]:

        img = self._img_graph.nodes[id]['data']
        img = self._cv_bridge.imgmsg_to_cv2(img)

        # compute homography
        G, duv, kp_img, kp_wrp, mask = self._feature_homography(img.astype(np.uint8), wrp.astype(np.uint8), return_kp=True)
        mean_pairwise_distance = None
        std_pairwise_distance = None
        n_matches = 0

        # wrp_pred = cv2.warpPerspective(img, G, (wrp.shape[1], wrp.shape[0]))
        # blend = yt_alpha_blend(wrp_pred, wrp)
        # cv2.imwrite('/tmp/blend.png', blend)

        # compute mpd

        if G is None:
            G = np.eye(3)

        if kp_img is not None and kp_wrp is not None:
            kp_img, kp_wrp = kp_img[mask == 1], kp_wrp[mask == 1]
            mean_pairwise_distance = np.linalg.norm(kp_img - kp_wrp, axis=1).mean()
            std_pairwise_distance = np.linalg.norm(kp_img - kp_wrp, axis=1).std()
            n_matches = kp_img.shape[0]
        return G, duv, mean_pairwise_distance, std_pairwise_distance, n_matches
