import os

import matplotlib.pyplot as plt
import numpy as np

import calibration_transform_coefficient as cd
from vel_2D_3C import Velocity2D_3C


def plot_window(x_label, rmse, plot=False):
    assert len(rmse_arr) == len(x_label), "rmse_arr should have same number of values as x_label."

    plt.figure(figsize=(10, 5))

    u = [l[0] for l in rmse]
    v = [l[1] for l in rmse]
    w = [l[2] for l in rmse]

    plt.plot(x_label, u, marker='o', label='X Component')  # RMSE for the X component
    plt.plot(x_label, v, marker='s', label='Y Component')  # RMSE for the Y component
    plt.plot(x_label, w, marker='^', label='Z Component')  # RMSE for the Z component

    plt.xlabel('Window Size')
    plt.ylabel('RMSE [mm]')
    plt.title('RMSE vs. Window Size for Each Component')
    plt.legend(loc='upper right')
    if plot:
        plt.show()


def rmse(dXYZ, truth):
    if len(truth.shape) > 1:
        x_diff = dXYZ[:, 0].flatten() - truth[:, 0].flatten()
        y_diff = dXYZ[:, 1].flatten() - truth[:, 1].flatten()
        z_diff = dXYZ[:, 2].flatten() - truth[:, 2].flatten()
    else:
        x_diff = dXYZ[:, 0].flatten() - truth[0]
        y_diff = dXYZ[:, 1].flatten() - truth[1]
        z_diff = dXYZ[:, 2].flatten() - truth[2]

    x_rmse = np.sqrt(np.mean(np.square(x_diff), axis=0))
    y_rmse = np.sqrt(np.mean(np.square(y_diff), axis=0))
    z_rmse = np.sqrt(np.mean(np.square(z_diff), axis=0))

    return [x_rmse, y_rmse, z_rmse]


def displace_rmse(rmse):
    x_labels = ["3", "5", "7", "9"]

    fig, ax = plt.subplots()
    x = range(len(rmse))

    u = [l[0] for l in rmse]
    v = [l[1] for l in rmse]
    w = [l[2] for l in rmse]

    ax.scatter(x, u, label='X', marker='o')
    ax.scatter(x, v, label='Y', marker='s')
    ax.scatter(x, w, label='Z', marker='^')

    ax.plot(x, u, 'o--', ms=2)
    ax.plot(x, v, 's--', ms=2)
    ax.plot(x, w, '^--', ms=2)

    ax.set_xlabel('Number of Calibration Set')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('RMSE [mm]')
    # Set legend location to 'upper right' and make room for it if it overlaps data
    ax.legend(loc='upper right')
    # plt.show()


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

# Translation
left_displacement = (left_pt[121:242, :] - left_pt[0:121, :])
right_displacement = (right_pt[121:242, :] - right_pt[0:121, :])


def plot_displacement():
    # Create a figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the X component of displacement
    ax1.quiver(left_pt[0:121, 0], left_pt[0:121, 1], left_displacement[0:121, 0], left_displacement[0:121, 1],
               color='b', scale=1, scale_units='xy', angles='xy', label='Left X')
    ax1.set_title('Left Camera')

    ax2.quiver(right_pt[0:121, 0], right_pt[0:121, 1], right_displacement[0:121, 0], right_displacement[0:121, 1],
               color='r', scale=1, scale_units='xy', angles='xy', label='Right Y')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Y position')
    ax2.set_title('Right Camera')

    # plt.show()


# plot_displacement()

x, y = np.meshgrid(np.arange(20, 400, 20), np.arange(20, 400, 20))
xyz = np.stack((x, y, np.zeros_like(x)), axis=-1)

xyz_p = xyz - [200, 200, 0]

# Apply the rotation
theta = np.deg2rad(3)  # Convert 3 degrees to radians
xp = xyz_p[:, :, 0] * np.cos(theta) + xyz_p[:, :, 1] * np.sin(theta)
yp = -xyz_p[:, :, 0] * np.sin(theta) + xyz_p[:, :, 1] * np.cos(theta)
zp = np.zeros_like(xp) + 2

rotated_xyz_p = np.stack((xp, yp, zp), axis=-1)
true_displace = rotated_xyz_p + [200, 200, 0] - xyz

rmse_arr = []

for x in range(2, 50, 2):
# for i in range(1, 5):
#     vel_object = Velocity2D_3C(left_pt[0:121, :], right_pt[0:121, :], left_displacement, right_displacement,
#                                os.path.join(path, f"Calibration/Set_{i}/left_cam_coeff.npy"),
#                                os.path.join(path, f"Calibration/Set_{i}/right_cam_coeff.npy"),
#                                window_size=44)
    vel_object = Velocity2D_3C(left_pt[0:121, :], right_pt[0:121, :], left_displacement, right_displacement,
                               os.path.join(path, f"Calibration/Set_3/left_cam_coeff.npy"),
                               os.path.join(path, f"Calibration/Set_3/right_cam_coeff.npy"),
                               window_size=x)
    displace_arr = vel_object.calculate_3D_displacement(xyz)
    grid, dXYZ_int = vel_object.interpolate_3D_displacement(xyz, displace_arr, False)

    truth = np.array((1.5, 1, 1.25))
    # _, truth = vel_object.interpolate_3D_displacement(xyz, true_displace)
    relative_error = (dXYZ_int - truth)
    # plot_interpolation(grid, relative_error, "Relative Error", plot=False)
    rmse_arr.append(rmse(dXYZ_int, truth))

x_values = list(range(2, 50, 2))
plot_window(x_values, rmse_arr)
# displace_rmse(np.array(rmse_arr))
plt.show()
