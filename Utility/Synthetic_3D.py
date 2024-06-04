import cv2
import numpy as np

# Set up the stereo camera parameters
camera_matrix_left = np.array([[500.0, 0.0, 320.0],
                              [0.0, 500.0, 240.0],
                              [0.0, 0.0, 1.0]])
camera_matrix_right = np.array([[500.0, 0.0, 320.0],
                               [0.0, 500.0, 240.0],
                               [0.0, 0.0, 1.0]])
dist_coeffs_left = np.array([0.0, 0.0, 0.0, 0.0])
dist_coeffs_right = np.array([0.0, 0.0, 0.0, 0.0])
R = np.eye(3)
T = np.array([[-0.1, 0.0, 0.0]])

# Create a blank image
width, height = 640, 480
img = np.zeros((height, width, 3), dtype=np.uint8)

# Undistort and rectify the images
left_undistorted = cv2.undistort(img, camera_matrix_left, dist_coeffs_left)
right_undistorted = cv2.undistort(img, camera_matrix_right, dist_coeffs_right)
left_rectified, right_rectified, Q = cv2.stereoRectify(camera_matrix_left, dist_coeffs_left,
                                                      camera_matrix_right, dist_coeffs_right,
                                                      (width, height), R, T)

# Combine the left and right images into a stereo pair
stereo_pair = np.hstack((left_rectified, right_rectified))

# Display the stereo pair
cv2.imshow('Stereo Pair', stereo_pair)
cv2.waitKey(0)
cv2.destroyAllWindows()