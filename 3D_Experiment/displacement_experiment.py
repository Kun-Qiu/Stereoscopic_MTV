import os

import numpy as np

import calibration_transform_coefficient as cd
import inverse_least_square as ils
from interpolate import DisplacementInterpolator
from vel_2D_3C import Velocity2D_3C

# Define paths
path = "../3D_Experiment/"
path_test = os.path.join(path, "Test")

# Create corner detection object
corner_detection_object = cd.CalibrationPointDetector(path_test, 10, (40, 40))
left_pos_path = os.path.join(path_test, "left_camera_pos.npy")
right_pos_path = os.path.join(path_test, "right_camera_pos.npy")

if not (os.path.exists(left_pos_path) and os.path.exists(right_pos_path)):
    left_pt = corner_detection_object.get_left_param()
    right_pt = corner_detection_object.get_right_param()
else:
    left_pt = np.load(left_pos_path, allow_pickle=True)
    right_pt = np.load(right_pos_path, allow_pickle=True)

left_pt = np.array(left_pt, dtype=np.float64)
right_pt = np.array(right_pt, dtype=np.float64)
left_displacement = left_pt[121:242, :] - left_pt[0:121, :]
right_displacement = right_pt[121:242, :] - right_pt[0:121, :]

x, y = np.meshgrid(np.arange(20, 400, 20), np.arange(20, 400, 20))
xyz = np.stack((x, y, np.zeros_like(x)), axis=-1)

vel_object = Velocity2D_3C(left_pt[0:121, :], right_pt[0:121, :], left_displacement, right_displacement,
                           os.path.join(path, "Calibration/left_cam_coeff.npy"),
                           os.path.join(path, "Calibration/right_cam_coeff.npy"))
displace_arr = vel_object.calculate_3D_displacement(xyz)
print("a")

# ## Interpolation For Left Camera ##
# left_intp_obj = DisplacementInterpolator(left_pt[0:121, :], left_displacement, grid_density=500)
# # left_intp_obj.plot_component_interpolation()
# x_left, y_left, dx_left, dy_left = left_intp_obj.get_interpolate()
#
# ## Interpolation For Right Camera ##
# right_intp_obj = DisplacementInterpolator(right_pt[0:121, :], right_displacement, grid_density=500)
# # right_magnitude = np.sqrt(np.sum(right_displacement ** 2, axis=1))
# # right_intp_obj.plot_interpolation()
# x_right, y_right, dx_right, dy_right = right_intp_obj.get_interpolate()
#
# ## Projection onto object plane ##
# left_coeff = np.load(os.path.join(path, "Calibration/left_cam_coeff.npy"))
# right_coeff = np.load(os.path.join(path, "Calibration/right_cam_coeff.npy"))
# inverse_object = ils.InverseTransform(left_coeff, right_coeff)

## Creating the desired grid to perform the extraction of displacement ##
# start = 20
# end = 400
# increment = 20
# x_values = np.arange(start, end, increment)
# y_values = np.arange(start, end, increment)
# x, y = np.meshgrid(x_values, y_values)
# interest_points = np.stack((x, y, np.zeros_like(x)), axis=-1)
# width, height, _ = interest_points.shape
# displacement_3D = np.zeros_like(interest_points)
# n = 10
#
#
# def displacement_3D_extractor(x_arr, y_arr, dx_arr, dy_arr, name):
#     displacement_arr = np.zeros((width, height, 2))
#     for i, column in enumerate(interest_points):
#         for j, point in enumerate(column):
#             camera_pt = inverse_object.projection_object_to_image(point, name)
#             distances = np.sqrt((x_arr - camera_pt[0]) ** 2 + (y_arr - camera_pt[1]) ** 2)
#             min_index = np.unravel_index(np.argmin(distances), distances.shape)
#
#             min_i, min_j = min_index
#             start_i = max(0, min_i - (n // 2))
#             end_i = min(distances.shape[0], min_i + (n // 2) + 1)
#             start_j = max(0, min_j - (n // 2))
#             end_j = min(distances.shape[1], min_j + (n // 2) + 1)
#
#             # Calculate average dx and dy within the window
#             dx_window = dx_arr[start_i:end_i, start_j:end_j]
#             dy_window = dy_arr[start_i:end_i, start_j:end_j]
#             valid_indices = ~np.isnan(dx_window) & ~np.isnan(dy_window)
#             if np.any(valid_indices):
#                 avg_dx = np.mean(dx_window[valid_indices])
#                 avg_dy = np.mean(dy_window[valid_indices])
#                 displacement_arr[i, j] = [avg_dx, avg_dy]
#
#     return displacement_arr
#
#
# right_camera_arr = displacement_3D_extractor(x_right, y_right, dx_right, dy_right, "right")
# right_camera_arr = right_camera_arr.reshape(-1, 2)
#
# left_camera_arr = displacement_3D_extractor(x_left, y_left, dx_left, dy_left, "left")
# left_camera_arr = left_camera_arr.reshape(-1, 2)
# interest_points = interest_points.reshape(-1, 3)

# for i in range(len(interest_points)):
#     xyz = interest_points[i]
#     left_np_pt = np.array((left_camera_arr[i, 0], left_camera_arr[i, 1]))
#     right_np_pt = np.array((right_camera_arr[i, 0], right_camera_arr[i, 1]))
#     result = inverse_object.inverse_displacement(xyz, left_np_pt, right_np_pt)
#     print(xyz, result)