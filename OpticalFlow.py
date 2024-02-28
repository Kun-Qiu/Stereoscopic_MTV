import cv2 as cv
import os
import matplotlib.pyplot as plt
from Utility.DeNoise import denoised_image

SOURCE_PATH = "Data//Source//"
TARGET_PATH = "Data//Target//"
TARGET_SYN_PATH = "Data/Synthetic Target//"

# Read source and target images
source_img = denoised_image(os.path.join(SOURCE_PATH, "source_25.png"))
target_img = denoised_image(os.path.join(TARGET_SYN_PATH, "img1.png"))

# Calculate optical flow
flow = cv.calcOpticalFlowFarneback(source_img, target_img, None,
                                   pyr_scale=0.5,
                                   levels=5,
                                   winsize=15,
                                   iterations=10,
                                   poly_n=7,
                                   poly_sigma=1.5,
                                   flags=None)

# Calculate magnitude and angle of optical flow vectors
magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

# Plot optical flow vectors on the source image
plt.imshow(cv.cvtColor(source_img, cv.COLOR_BGR2RGB))
plt.quiver(range(0, flow.shape[1], 10),
           range(0, flow.shape[0], 10),
           flow[::10, ::10, 0],  # u component of flow
           flow[::10, ::10, 1],  # v component of flow (invert y-axis)
           magnitude[::10, ::10],  # magnitude of flow
           angles='xy', scale_units='xy', scale=1, cmap='viridis')
plt.colorbar()  # Add color bar to indicate magnitude
plt.show()


