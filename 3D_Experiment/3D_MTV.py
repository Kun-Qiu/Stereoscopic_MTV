import os
import numpy as np
import inverse_least_square as ils
import calibration_corner_detection as cd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def projection_error_display(true_pts, predicted_pts, points_per_plot=121):
    num_points = len(true_pts)

    fig = plt.figure(figsize=(12, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=2)

    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[1, 0])
    ax5 = fig.add_subplot(spec[1, 1])
    ax6 = fig.add_subplot(spec[1, 2])

    all_errors = []
    for i in range(num_points):
        error = np.sqrt(np.sum(predicted_pts[i] - true_pts[i]) ** 2)
        all_errors.append(error)

        plot_index = i // points_per_plot

        if plot_index == 0:
            ax = ax1
        elif plot_index == 1:
            ax = ax2
        elif plot_index == 2:
            ax = ax3
        elif plot_index == 3:
            ax = ax4
        elif plot_index == 4:
            ax = ax5
        elif plot_index == 5:
            ax = ax6
        else:
            continue

        ax.scatter(i % points_per_plot, error, c='k',
                   label='Projection Error' if (i % points_per_plot) == 0 else "")

        index = 0 + plot_index * points_per_plot
        ax.set_title(f'Position: ({true_pts[index][0]}, {true_pts[index][1]}, {true_pts[index][2]})')
        ax.set_xlabel('Index')
        ax.set_ylabel('Error [Pixel]')

    fig.tight_layout()
    plt.show()


path = '../3D_Experiment/'
left_coeff = np.load(os.path.join(path, "Calibration/left_cam_coeff.npy"))
right_coeff = np.load(os.path.join(path, "Calibration/right_cam_coeff.npy"))
inverse_object = ils.InverseTransform(left_coeff, right_coeff)

corner_detection_object = cd.CornerDetector(os.path.join(path, "Test"), 10, (40, 40))

dx = [-3, -3, -3, 0, 2, 3]
dy = [3, 3, 0, -1, 3, 3]

path_test = os.path.join(path, "Test")
if not (os.path.exists(os.path.join(path_test, "left_camera_pos.npy")) and
        os.path.exists(os.path.join(path_test, "right_camera_pos.npy")) and
        os.path.exists(os.path.join(path_test, "3D_camera_pos.npy"))):

    true_pt = corner_detection_object.get_initial_calibrate_corners(dx=dx, dy=dy)
    initial_pt = corner_detection_object.get_initial_calibrate_corners()
    left_pt = corner_detection_object.get_left_corners()
    right_pt = corner_detection_object.get_right_corners()
else:
    initial_pt = np.load(os.path.join(path_test, "3D_camera_pos.npy"), allow_pickle=True)
    left_pt = np.load(os.path.join(path_test, "left_camera_pos.npy"), allow_pickle=True)
    right_pt = np.load(os.path.join(path_test, "right_camera_pos.npy"), allow_pickle=True)
    true_pt = np.load(os.path.join(path_test, "3D_true_camera_pos.npy"), allow_pickle=True)

predicted_3D_pos = []
for point_left, point_right in zip(left_pt, right_pt):
    predicted_3D_pos.append(inverse_object.inverse_least_square(point_left, point_right))

predicted_3D_pos = np.array(predicted_3D_pos)
projection_error_display(np.array(true_pt), np.array(predicted_3D_pos))
