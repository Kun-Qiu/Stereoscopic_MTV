import cv2 as cv
import matplotlib.pyplot as plt
from Utility.DeNoise import denoised_image


class OpticalFlow:
    def __init__(self, source_path, target_path):
        self._SOURCE_PATH = source_path
        self._TARGET_PATH = target_path
        self._flow = None

    def calculate_optical_flow(self):
        # Read source and target images
        source_img = denoised_image(self._SOURCE_PATH)
        target_img = denoised_image(self._TARGET_PATH)

        # Calculate optical flow
        self._flow = cv.calcOpticalFlowFarneback(source_img, target_img, None,
                                                 pyr_scale=0.5,
                                                 levels=5,
                                                 winsize=15,
                                                 iterations=10,
                                                 poly_n=7,
                                                 poly_sigma=1.5,
                                                 flags=0)

    def visualize_flow(self):
        source_img = cv.imread(self._SOURCE_PATH)
        magnitude, angle = cv.cartToPolar(self._flow[..., 0],
                                          self._flow[..., 1])

        # Plot optical flow vectors on the source image
        plt.imshow(cv.cvtColor(source_img, cv.COLOR_BGR2RGB))
        plt.quiver(range(0, self._flow.shape[1], 10),
                   range(0, self._flow.shape[0], 10),
                   self._flow[::10, ::10, 0],  # u component of flow
                   self._flow[::10, ::10, 1],  # v component of flow (invert y-axis)
                   magnitude[::10, ::10],  # magnitude of flow
                   angles='xy', scale_units='xy', scale=1, cmap='viridis')
        plt.colorbar()  # Add color bar to indicate magnitude
        plt.show()

    def get_flow(self):
        return self._flow
