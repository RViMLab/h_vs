import numpy as np
import cv2

from homography_generators.base_homography_generator import BaseHomographyGenerator

class CalibrationPatternHomographyGenerator(BaseHomographyGenerator):
    def __init__(self, K: np.array, d: np.array, undistort: bool=False):
        super().__init__(K=K, d=d, buffer_size=1, undistort=undistort)

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
        if found0 and found:
            G, _ = cv2.findHomography(pts0, pts, cv2.RANSAC)

        return G

    def _undistort(self, img):
        return cv2.undistort(img, self._K, self._d)
