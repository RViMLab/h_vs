from abc import ABC


from homography_generators.base_homography_generator import BaseHomographyGenerator

class EndoscopyHomographyGenerator(BaseHomographyGenerator):
    def __init__(self, K: np.array, D: np.array, buffer_size: int, undistort: bool=False):
        super().__init__(K, D, buffer_size, undistort)
        pass
