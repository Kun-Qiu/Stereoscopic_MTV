import os
import matplotlib.pyplot as plt
import numpy as np

import calibration_transform_coefficient as cd
from utility.Visualization import plot_interpolation
from vel_2D_3C import Velocity2D_3C


# Define paths
base_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026"
calibration_path = os.path.join(base_path, "Calibration_Volume")
test_path = os.path.join(base_path, "Test_Cases")
save_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res"

############### Experimental Cases ###############
## Reference Image
ref_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\ref"
ref_l_path = os.path.join(ref_path, "left")
ref_r_path = os.path.join(ref_path, "right")
ref_obj = cd.CalibrationPointDetector(
    ref_l_path, ref_r_path, save_path, 10, (20, 20)
    )

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
case_path = r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\in_plane"
test_l_path = os.path.join(case_path, "left")
test_r_path = os.path.join(case_path, "right")
case_obj = cd.CalibrationPointDetector(
    test_l_path, test_r_path, save_path, 10, (20, 20)
    )

# Obatin position in image plane of left and right camera points
# test_left_pt = np.array(case_obj.get_left_param("left_in_plane"), dtype=np.float64)
# test_right_pt = np.array(case_obj.get_right_param("right_in_plane"), dtype=np.float64)

test_left_pt = np.array(
    np.load(
        r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\left_in_plane.npy",
        allow_pickle=True), 
    dtype=np.float64
    )
test_right_pt = np.array(
    np.load(
        r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\res\right_in_plane.npy",
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
ax1.set_xlabel('X position')
ax1.set_ylabel('Y position')
ax1.set_title('Left Camera')

ax2.quiver(
    ref_right_pt[..., 0].ravel(), ref_right_pt[..., 1].ravel(), 
    right_displacement[..., 0].ravel(), right_displacement[..., 1].ravel(),
    color='r', scale=0.5, scale_units='xy', angles='xy', label='Right Y'
    )
ax2.set_xlabel('X position')
ax2.set_title('Right Camera')

############ Initialize the Ground Truth Measurements ################

# Default xyz is the calibration target dimensions
calibration_dim = 400   # 400 mm on each side 
square_dim = 20
x, y = np.meshgrid(
    np.arange(0, calibration_dim+square_dim, square_dim), 
    np.arange(0, calibration_dim+square_dim, square_dim)
    )
xyz = np.stack((x, y, np.zeros_like(x)), axis=-1)

# In-Plane Rotational Field Ground Truth
rotation = 2 # degrees
xyz_p = xyz - [calibration_dim/2, calibration_dim/2, 0]  # Center the coordinates

theta = np.deg2rad(rotation) 
xp = xyz_p[:, :, 0] * np.cos(theta) + xyz_p[:, :, 1] * np.sin(theta)
yp = -xyz_p[:, :, 0] * np.sin(theta) + xyz_p[:, :, 1] * np.cos(theta)
zp = np.zeros_like(xp)

in_plane_disp = np.stack((xp, yp, zp), axis=-1) - xyz_p 

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

# Compute absolute relative error safely
with np.errstate(divide='ignore', invalid='ignore'):
    relative_error = np.abs(
        (displace_3d[..., 0:2] - in_plane_disp[..., 0:2]) / in_plane_disp[..., 0:2] * 100
        )
    zero_z = np.zeros((relative_error.shape[0], relative_error.shape[1], 1))

    mask = np.ones_like(relative_error, dtype=bool)
    mask[..., 0] &= (xyz[..., 1] != 200)
    mask[..., 1] &= (xyz[..., 0] != 200)

    relative_error[~mask] = np.nan
    relative_error = np.concatenate((relative_error, zero_z), axis=2)

# grid, dXYZ_int = vel_object.interpolate_3D_displacement(xyz, displace_3d)
# plot_interpolation(grid, dXYZ_int, "Displacement", "mm", path=r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\in_plane")


grid_rel, dXYZ_rel = vel_object.interpolate_3D_displacement(xyz, relative_error)
plot_interpolation(grid_rel, dXYZ_rel, "Relative Error", "%", path=r"C:\Users\Kun Qiu\Desktop\Thesis_2026\Test_Cases\in_plane")