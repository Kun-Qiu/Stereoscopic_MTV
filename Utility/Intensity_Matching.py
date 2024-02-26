import cv2
from skimage.exposure import match_histograms
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to Data
SOURCE_PATH = "..//Data//Source//"
TARGET_PATH = "..//Data//Target//"


def intensity_matching(source, target):
    if source is None:
        print("Error: Failed to load source image.")
        return None
    if target is None:
        print("Error: Failed to load target image.")
        return None

    source = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2GRAY)

    # Matching the histogram of target to source (Grayscale)
    match = match_histograms(target, source, channel_axis=None)
    return match


def intensity_disparity(source, target):
    source = cv2.imread(source)
    target = cv2.imread(target)

    print(target.shape)

    # Convert images to grayscale if not
    if source.shape[2] != 1:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if target.shape[2] != 1:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # Calculate the difference in intensity
    disparity = source.astype(np.float32) - target.astype(np.float32)

    plt.figure(figsize=(8, 6))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.title('Intensity Difference Map')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()
    return None


input_img = os.path.join(SOURCE_PATH, 'source.png')
target_img = os.path.join(TARGET_PATH, 'target.png')
matched_img = os.path.join(TARGET_PATH, 'matched_image.png')

# Perform histogram matching
# matched_img = intensity_matching(input_img, target_img)

# Calculate intensity difference
intensity_disparity(target_img, matched_img)

# Display the matched image using OpenCV
# if matched_img is not None:
    # cv2.imwrite(os.path.join(TARGET_PATH, "matched_image.png"), matched_img)
