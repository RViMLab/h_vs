import numpy as np
import cv2

from homography_generators.base_homography_generator import BaseHomographyGenerator

class CalibrationPatternHomographyGenerator(BaseHomographyGenerator):
    def __init__(self, K: np.array, D: np.array, undistort: bool=False):
        super().__init__(K, D, buffer_size=1, undistort=undistort)

    def desiredHomography(self, img0, patternSize=(4, 11)):

        img = self._imgs[0]

        # convert to grayscale
        img0 = cv2.cvtColor(img0.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.uint8)
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # find points
        found0, pts0 = cv2.findCirclesGrid(img0, patternSize=patternSize, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        found, pts = cv2.findCirclesGrid(img, patternSize=patternSize, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        # compute homography
        G = np.eye(3)
        mean_pairwise_distance = None
        if found0 and found:
            G, _ = cv2.findHomography(pts0, pts, cv2.RANSAC)

            # evaluate in normalized image coordinates
            # K^(-1)*[(pts,1) - (pts0,1)]
            K_inv = np.linalg.inv(self._K)

            pts  = np.concatenate([pts, np.ones((pts.shape[0], pts.shape[1], 1))], axis=2)
            pts0 = np.concatenate([pts0, np.ones((pts0.shape[0], pts0.shape[1], 1))], axis=2)

            # Nx1x3 -> Nx3x1
            pts  = pts.transpose(0, 2, 1)
            pts0 = pts0.transpose(0, 2, 1)

            pts  = np.matmul(K_inv, pts)
            pts0 = np.matmul(K_inv, pts0)

            mean_pairwise_distance = np.linalg.norm(pts[:,:-1] - pts0[:,:-1], axis=1).mean()

        return G, mean_pairwise_distance
