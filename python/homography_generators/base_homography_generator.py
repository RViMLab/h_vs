from abc import ABC, abstractmethod
import numpy as np


class BaseHomographyGenerator(ABC):
    def __init__(self, K: np.array, d: np.array, buffer_size: int, undistort=False: bool):
        self._imgs = []
        self._K = K
        self._d = d
        self._buffer_size = buffer_size
        self._ud = undistort

    def addImg(self, img):
        """Append image buffer by img and undistort.
        """
        if self._ud:
            img = self._undistort(img)
        if len(self.imgs) >= self.buffer_size:
            self.imgs.pop(0)
        self.imgs.append(img)

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

    @abstractmethod
    def _undistort(self, img):
        """Undistord img.

        returns: img, undistorted image
        """
        return
