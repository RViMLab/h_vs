import numpy as np
import cv2
from typing import Tuple
import networkx as nx

from homography_generators.base_homography_generator import BaseHomographyGenerator
from homography_generators.homography_imitation_learning import utils


class StoredViewHomographyGenerator(BaseHomographyGenerator):
    def __init__(self, K: np.ndarray, D: np.ndarray, undistort: bool=False) -> None:
        super().__init__(K, D, buffer_size=None, undistort=undistort)

        # self._feature_detector = cv2.xfeatures.SIFT_create()  # proprietary, how to use?
        # self._feature_detector = cv2.xfeatures.SURF_create()  # proprietary, how to use?
        self._feature_detector = cv2.ORB_create(nfeatures=2000)
        # self._feature_detector = cv2.FastFeatureDetector_create()

        self._feature_homography = utils.FeatureHomographyEstimation(self._feature_detector)

    def desiredHomography(self, wrp: np.ndarray, id: int) -> Tuple[np.ndarray, np.ndarray]:

        img = self._img_graph.nodes[id]['data']

        if self._ud:
            img0, _ = self.undistort(img0)

        # compute homography
        G, _ = self._feature_homography(img, wrp)
        mean_pairwise_distance = None

        if G is None:
            G = np.eye(3)

        # if found0 and found:
        #     G, _ = cv2.findHomography(pts0, pts, cv2.RANSAC)

        #     # evaluate in normalized image coordinates
        #     # K^(-1)*[(pts,1) - (pts0,1)]
        #     K_inv = np.linalg.inv(self._K)

        #     pts  = np.concatenate([pts, np.ones((pts.shape[0], pts.shape[1], 1))], axis=2)
        #     pts0 = np.concatenate([pts0, np.ones((pts0.shape[0], pts0.shape[1], 1))], axis=2)

        #     # Nx1x3 -> Nx3x1
        #     pts  = pts.transpose(0, 2, 1)
        #     pts0 = pts0.transpose(0, 2, 1)

        #     pts  = np.matmul(K_inv, pts)
        #     pts0 = np.matmul(K_inv, pts0)

        #     mean_pairwise_distance = np.linalg.norm(pts[:,:-1] - pts0[:,:-1], axis=1).mean()

        return G, mean_pairwise_distance
