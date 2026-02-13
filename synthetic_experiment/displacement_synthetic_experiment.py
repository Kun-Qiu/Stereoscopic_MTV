import os
import matplotlib.pyplot as plt
import numpy as np

import calibration_transform_coefficient as cd
from utility.Visualization import plot_interpolation
from vel_2D_3C import Velocity2D_3C


# Define paths
base_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026"
calibration_path = os.path.join(base_path, "Calibration_Volume_Dense")
test_path = os.path.join(base_path, "Test_Cases")
save_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res"
rotation = 1 # degrees
plane = "out_plane"

############### Experimental Cases ###############
## Reference Image
ref_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\ref"
ref_l_path = os.path.join(ref_path, "left")
ref_r_path = os.path.join(ref_path, "right")
ref_obj = cd.CalibrationPointDetector(
    ref_l_path, ref_r_path, save_path, 10, (20, 20)
    )

# ref_left_pt = np.array(ref_obj.get_left_param(f"ref_left"), dtype=np.float64)
# ref_right_pt = np.array(ref_obj.get_right_param(f"ref_right"), dtype=np.float64)
ref_left_pt = np.array(
    np.load(
        r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\ref_left.npy",
        allow_pickle=True), 
    dtype=np.float64
    )
ref_right_pt = np.array(
    np.load(
        r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\ref_right.npy",
        allow_pickle=True), 
    dtype=np.float64
    )

## Case 1: Displacement due in plane rotation only
case_path = fr"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\{plane}"
test_l_path = os.path.join(case_path, "left")
test_r_path = os.path.join(case_path, "right")
case_obj = cd.CalibrationPointDetector(
    test_l_path, test_r_path, save_path, 10, (20, 20)
    )

# Obatin position in image plane of left and right camera points
# test_left_pt = np.array(case_obj.get_left_param(f"left_{plane}"), dtype=np.float64)
# test_right_pt = np.array(case_obj.get_right_param(f"right_{plane}"), dtype=np.float64)
test_left_pt = np.array(
    np.load(
        fr"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\left_{plane}.npy",
        allow_pickle=True), 
    dtype=np.float64
    )
test_right_pt = np.array(
    np.load(
        fr"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\right_{plane}.npy",
        allow_pickle=True), 
    dtype=np.float64
    )

# Displacement Calculations
left_displacement = (test_left_pt - ref_left_pt)
right_displacement = (test_right_pt - ref_right_pt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.quiver(
    ref_left_pt[..., 0].ravel(), ref_left_pt[..., 1].ravel(), 
    left_displacement[..., 0].ravel(), left_displacement[..., 1].ravel(),
    color='b', scale=0.5, scale_units='xy', angles='xy', label='Left X'
    )
ax1.set_xlabel('X position', fontsize=14)
ax1.set_ylabel('Y position', fontsize=14)
ax1.set_title('Left Camera', fontsize=14)

ax2.quiver(
    ref_right_pt[..., 0].ravel(), ref_right_pt[..., 1].ravel(), 
    right_displacement[..., 0].ravel(), right_displacement[..., 1].ravel(),
    color='r', scale=0.5, scale_units='xy', angles='xy', label='Right Y'
    )
ax2.set_xlabel('X position', fontsize=14)
ax2.set_title('Right Camera', fontsize=14)

############ Initialize the Ground Truth Measurements ################

# Default xyz is the calibration target dimensions
calibration_dim = 400   # 400 mm on each side 
square_dim = 10
x, y = np.meshgrid(
    np.arange(0, calibration_dim+square_dim, square_dim), 
    np.arange(0, calibration_dim+square_dim, square_dim)
    )
xyz = np.stack((x, y, np.zeros_like(x)), axis=-1)
xyz_p = xyz - [calibration_dim/2, calibration_dim/2, 0]  # Center the coordinates
theta = np.deg2rad(rotation) 

# In-Plane Rotational Field Ground Truth
# xp = xyz_p[:, :, 0] * np.cos(theta) + xyz_p[:, :, 1] * np.sin(theta)
# yp = -xyz_p[:, :, 0] * np.sin(theta) + xyz_p[:, :, 1] * np.cos(theta)
# zp = xyz_p[:, :, 2]  

# Out-Plane Rotational with translation
xp =  (xyz_p[:, :, 0] * np.cos(theta) - xyz_p[:, :, 2] * np.sin(theta)) + 1
yp =  xyz_p[:, :, 1]                             
zp =  xyz_p[:, :, 0] * np.sin(theta) + xyz_p[:, :, 2] * np.cos(theta)

# Pure Translational in all directions  (-1 mm , -2.5 mm, 2 mm) 
# xp =  xyz_p[:, :, 0] - 1
# yp =  xyz_p[:, :, 1] - 2.5                         
# zp =  xyz_p[:, :, 2] + 2

truth_disp = (np.stack((xp, yp, zp), axis=-1) - xyz_p)

############ Initialize the Velocity2D_3C object ################
left_cam_coeff_path = os.path.join(calibration_path, "left_cam_coeff.npy")
right_cam_coeff_path = os.path.join(calibration_path, "right_cam_coeff.npy")
vel_object = Velocity2D_3C(
    ref_left_pt, ref_right_pt, 
    left_displacement, right_displacement,
    left_cam_coeff_path, right_cam_coeff_path
    )

# Calculate 3D displacement
displace_3d = vel_object.calculate_3D_displacement(xyz)  # shape: (Nx, Ny, 3)

relative_error = np.full_like(displace_3d, np.nan)  # initialize with NaNs
valid_mask = truth_disp != 0
relative_error[valid_mask] = np.abs(
    (displace_3d[valid_mask] - truth_disp[valid_mask]) / truth_disp[valid_mask] * 100
    )

fig_path = fr"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\{plane}"

grid, dXYZ_int = vel_object.interpolate_3D_displacement(xyz, displace_3d)
plot_interpolation(grid, dXYZ_int, "Displacement", "mm", path=fig_path)

grid_rel, dXYZ_rel = vel_object.interpolate_3D_displacement(xyz, relative_error)
plot_interpolation(grid_rel, dXYZ_rel, "Relative Error", "%", path=fig_path)