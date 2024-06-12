import cv2
import numpy as np
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
        self._flow = cv2.calcOpticalFlowFarneback(source_img, target_img, None,
                                                  pyr_scale=0.5,
                                                  levels=3,
                                                  winsize=15,
                                                  iterations=10,
                                                  poly_n=10,
                                                  poly_sigma=1.5,
                                                  flags=0)

    def visualize_flow(self):
        source_img = cv2.imread(self._SOURCE_PATH)
        magnitude, angle = cv2.cartToPolar(self._flow[..., 0],
                                           self._flow[..., 1])

        # Plot optical flow vectors on the source image
        plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
        plt.quiver(range(0, self._flow.shape[0], 10),
                   range(0, self._flow.shape[1], 10),
                   self._flow[::10, ::10, 0],  # u component of flow
                   self._flow[::10, ::10, 1],  # v component of flow
                   magnitude[::10, ::10],  # magnitude of flow
                   angles='xy', scale_units='xy', scale=1, cmap='viridis')
        plt.colorbar()
        plt.show()

    def get_flow(self):
        return np.stack((self._flow[:, :, 0], self._flow[:, :, 1]), axis=-1)


source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Target/frame_0_2us.png'
optical_flow_object = OpticalFlow(source_path, target_path)
optical_flow_object.calculate_optical_flow()
optical_flow_object.visualize_flow()
