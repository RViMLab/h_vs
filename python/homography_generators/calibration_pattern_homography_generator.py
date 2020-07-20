from homography_generators.base_homography_generator import BaseHomographyGenerator

class CalibrationPatternHomographyGenerator(BaseHomographyGenerator):
    def desiredHomography(self):
        return np.array([1.])

    def _undistort(self, img):
        return img
