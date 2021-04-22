import numpy as np
import networkx as nx
import cv2
from typing import List, Tuple
from abc import ABC, abstractmethod

class BaseHomographyGenerator(ABC):
    def __init__(self, K: np.ndarray, D: np.ndarray, buffer_size: int, undistort: bool=False) -> None:
        self._img_graph = nx.DiGraph()
        self._current_id = -1
        self._buffer_size = buffer_size
        self._ud = undistort

        self._K = K
        self._D = D

    def addImg(self, img: np.ndarray) -> None:
        r"""Append image graph by img and undistort if desired.
        """
        self._current_id += 1
        if self._ud:
            img, _ = self.undistort(img)

        self._img_graph.add_node(self._current_id, data=img)
        if self._current_id is not 0:
            self._img_graph.add_edge(self._current_id - 1, self._current_id)

        if len(self._img_graph) > self._buffer_size:
            first = list(nx.topological_sort(self._img_graph))[0]
            self._img_graph.remove_node(first)
            self._img_graph = nx.relabel_nodes(self._img_graph, lambda x: x - 1)
            self._current_id -= 1


    def clearImgGraph(self) -> None:
        r"""Clear image buffer.
        """
        self._current_id = -1
        self._img_graph.clear()

    @abstractmethod
    def desiredHomography(self) -> np.ndarray:
        r"""Compute desired homography based on image buffer.

        Return: 
            G (np.ndarray): projective homography
        """
        return


    def undistort(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Undistord img.
        
        Args: 
            img (np.ndarray): Image in OpenCV convention of size HxWxC

        Return: 
            (img, K) (Tuple[np.ndarray, np.ndarray]): Tuple containing undistorted image and new camera matrix
        """
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w,h), 1, (w,h))  # alpha set to 1.

        return cv2.undistort(img, self._K, self._D, None, newcameramtx), newcameramtx

    @property
    def ImgGraph(self) -> nx.DiGraph:
        r"""Image buffer.
        """
        return self._img_graph

    @ImgGraph.deleter
    def ImgGraph(self) -> None:
        self.clearImgGraph()

    @property
    def K(self) -> np.ndarray:
        r"""Camera intrinsics.
        """
        return self._K

    @K.setter
    def K(self, value: np.ndarray) -> None:
        self._K = value

    @property
    def D(self) -> np.ndarray:
        r"""Camera distortion.
        """
        return self._D

    @D.setter
    def D(self, value: np.ndarray) -> None:
        self._D = value
