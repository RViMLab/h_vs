import numpy as np
import networkx as nx
import cv2
from typing import List, Tuple
from abc import ABC, abstractmethod

class BaseHomographyGenerator(ABC):
    def __init__(self, K: np.ndarray, D: np.ndarray, buffer_size: int=None, undistort: bool=False) -> None:
        self._img_graph = nx.Graph()
        self._current_id = None
        self._prev_id = self._current_id
        self._buffer_size = buffer_size
        self._ud = undistort

        self._K = K
        self._D = D

    def addImg(self, img: np.ndarray) -> int:
        r"""Append image graph by img and undistort if desired.
        """
        self._prev_id = self._current_id
        self._current_id = len(self._img_graph)
        if self._ud:
            img, _ = self.undistort(img)

        self._img_graph.add_node(self._current_id, data=img)
        if self._prev_id is not None:
            self._img_graph.add_edge(self._prev_id, self._current_id)
            self._img_graph.add_edge(self._current_id, self._prev_id)  # bi-directional graph

        if self._buffer_size is not None:
            if len(self._img_graph) > self._buffer_size:
                first = list(nx.topological_sort(self._img_graph))[0]
                self._img_graph.remove_node(first)
                self._img_graph = nx.relabel_nodes(self._img_graph, lambda x: x - 1)
                self._current_id -= 1

        return self._current_id

    def clearImgGraph(self) -> None:
        r"""Clear image buffer.
        """
        self._current_id = None
        self._prev_id = self._current_id
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
    def ImgGraph(self) -> nx.Graph:
        r"""Image buffer.
        """
        return self._img_graph

    @ImgGraph.deleter
    def ImgGraph(self) -> None:
        self.clearImgGraph()

    @property
    def ID(self) -> int:
        r"""Get current id.
        """
        return self._current_id

    @ID.setter
    def ID(self, id: int) -> None:
        r"""Set current node.
        """
        self._current_id = id

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    class DummHomographyGenerator(BaseHomographyGenerator):
        def desiredHomography():
            return np.eye(3)
    
    def visualize(G: nx.DiGraph):
        nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
        H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

        nx.draw(H)
        plt.show()


    K = np.eye(5)
    D = np.zeros(6)
    hg = DummHomographyGenerator(K, D)

    # add 3 images to graph
    img = np.ones((16, 16))

    id = hg.addImg(img)
    print('Added node: {}'.format(id))
    id = hg.addImg(img)
    print('Added node: {}'.format(id))
    id =hg.addImg(img)
    print('Added node: {}'.format(id))

    graph = hg.ImgGraph
    visualize(graph)

    # move id
    print('Current id: {}'.format(hg.ID))
    id = 1
    print('Setting id to: {}'.format(id))
    hg.ID = id
    print('Current id: {}'.format(hg.ID))

    # add new image
    id = hg.addImg(img)
    print('Added node: {}'.format(id))

    graph = hg.ImgGraph
    visualize(graph)

    # find path
    src = hg.ID
    dst = 0

    path = nx.dijkstra_path(hg.ImgGraph, src, dst)
    print('Path from {} to {}: {}'.format(src, dst, path))

    # clear graph
    hg.clearImgGraph()

    # find path
    src = hg.ID
    dst = 0

    try:
        path = nx.dijkstra_path(hg.ImgGraph, src, dst)
    except:
        print('Path could not be found')
