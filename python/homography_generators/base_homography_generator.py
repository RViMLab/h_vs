import numpy as np
import cv2
from abc import ABC, abstractmethod

class BaseHomographyGenerator(ABC):
    def __init__(self, K: np.array, D: np.array, buffer_size: int, undistort: bool=False):
        self._imgs = []
        self._buffer_size = buffer_size
        self._ud = undistort

        self._K = K
        self._D = D

    def addImg(self, img: np.array):
        """Append image buffer by img and undistort if desired.
        """
        if self._ud:
            img = self._undistort(img)
        if len(self._imgs) >= self._buffer_size:
            self._imgs.pop(0)
        self._imgs.append(img)

    def clearBuffer(self):
        """Clear image buffer.
        """
        self._imgs.clear()

    @abstractmethod
    def desiredHomography(self) -> np.array:
        """Compute desired homography based on image buffer.

        returns: G, projective homography (np.array)
        """
        return

    def _undistort(self, img: np.array):
        """Undistord img.
        param: img, image in OpenCV convention of size HxWxC

        returns: img, undistorted image
        """
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w,h), 1, (w,h))

        return cv2.undistort(img, self._K, self._D, None, newcameramtx)

    @property
    def Imgs(self):
        """Image buffer.
        """
        return self._imgs

    @Imgs.deleter
    def Imgs(self):
        self.clearBuffer()

    @property
    def K(self):
        """Camera intrinsics.
        """
        return self._K

    @K.setter
    def K(self, value: np.array):
        self._K = value

    @property
    def D(self):
        """Camera distortion.
        """
        return self._D

    @D.setter
    def D(self, value: np.array):
        self._D = value
