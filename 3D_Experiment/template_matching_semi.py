import os
import numpy as np
import inverse_least_square as ils
import calibration_corner_detection as cd
import matplotlib.pyplot as plt
from interpolate import DisplacementInterpolator

# Define paths
path = '../3D_Experiment/'
path_test = os.path.join(path, "Velocity Test")

# Create corner detection object
corner_detection_object = cd.CornerDetector(path_test, 5, (40, 40))
left_pos_path = os.path.join(path_test, "left_camera_pos.npy")
right_pos_path = os.path.join(path_test, "right_camera_pos.npy")

if not (os.path.exists(left_pos_path) and os.path.exists(right_pos_path)):
    left_pt = corner_detection_object.get_left_corners()
    right_pt = corner_detection_object.get_right_corners()
else:
    left_pt = np.load(left_pos_path, allow_pickle=True)
    right_pt = np.load(right_pos_path, allow_pickle=True)

# Ensure the data is numeric
left_pt = np.array(left_pt, dtype=np.float64)
right_pt = np.array(right_pt, dtype=np.float64)

# Extract initial and final points
left_init = left_pt[0:36, :]
left_final = left_pt[36:72, :]
right_init = right_pt[0:36, :]
right_final = right_pt[36:72, :]

# Compute displacement
left_displacement = left_final - left_init
right_displacement = right_final - right_init

# Interpolation setup
left_x_coords = left_init[:, 0]
left_y_coords = left_init[:, 1]
left_u_displacement = left_displacement[:, 0]
left_v_displacement = left_displacement[:, 1]

right_x_coords = right_init[:, 0]
right_y_coords = right_init[:, 1]
right_u_displacement = right_displacement[:, 0]
right_v_displacement = right_displacement[:, 1]

## Interpolation For Left Camera ##
points_left = np.stack((left_x_coords, left_y_coords), axis=1)
displacement_left = np.stack((left_u_displacement, left_v_displacement), axis=1)
left_intp_obj = DisplacementInterpolator(points_left, displacement_left, grid_density=2000)
x_left, y_left, dx_left, dy_left = left_intp_obj.get_interpolate()

## Interpolation For Right Camera ##
points_right = np.stack((right_x_coords, right_y_coords), axis=1)
displacement_right = np.stack((right_u_displacement, right_v_displacement), axis=1)
right_intp_obj = DisplacementInterpolator(points_right, displacement_right, grid_density=2000)
x_right, y_right, dx_right, dy_right = right_intp_obj.get_interpolate()

## Projection onto Object Plane ##
left_coeff_path = os.path.join(path, "Calibration/left_cam_coeff.npy")
right_coeff_path = os.path.join(path, "Calibration/right_cam_coeff.npy")

# Load calibration coefficients
left_coeff = np.load(left_coeff_path)
right_coeff = np.load(right_coeff_path)
inverse_object = ils.InverseTransform(left_coeff, right_coeff, i=3, j=3, k=2)

## Creating the desired grid to perform the extraction of displacement ##
start = 0
end = 400
increment = 40
x_values = np.arange(start, end + increment, increment)
y_values = np.arange(start, end + increment, increment)
x, y = np.meshgrid(x_values, y_values)
interest_points = np.stack((x, y, np.zeros_like(x)), axis=-1)
displacement_3D = np.zeros_like(interest_points)

for column in interest_points:
    for point in column:
        left_camera_pt = inverse_object.projection_object_to_image(point, "left")
        right_camera_pt = inverse_object.projection_object_to_image(point, "right")
        print(left_camera_pt, right_camera_pt)
