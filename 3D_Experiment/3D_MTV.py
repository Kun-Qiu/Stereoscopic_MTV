import os
import numpy as np
import inverse_least_square as ils
import calibration_corner_detection as cd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def projection_error_display(true_pts, predicted_pts, points_per_plot=121, filter_outlier=True):
    num_points = len(true_pts)

    fig = plt.figure(figsize=(12, 8))
    spec = gridspec.GridSpec(ncols=3, nrows=2)

    axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(3)]

    all_errors = np.sqrt(np.sum((predicted_pts - true_pts) ** 2, axis=1))

    num_plots = num_points // points_per_plot
    outlier_count = 0

    for plot_index in range(num_plots):
        start_idx = plot_index * points_per_plot
        end_idx = (plot_index + 1) * points_per_plot
        subset_errors = all_errors[start_idx:end_idx]

        if filter_outlier:
            median_error = np.median(subset_errors)
            mad = np.median(np.abs(subset_errors - median_error))
            threshold = 4.0 * 1.4826 * mad
            filtered_indices = np.where(np.abs(subset_errors - median_error) <= threshold)[0]
            outlier_count += (points_per_plot - len(filtered_indices))
        else:
            filtered_indices = np.arange(len(subset_errors))

        ax = axes[plot_index]

        for idx in filtered_indices:
            ax.scatter(start_idx + idx, subset_errors[idx], c='k',
                       label='Projection Error' if (start_idx + idx) == 0 else "")

        ax.set_title(
            f'Position: ({int(np.average(true_pts[start_idx:end_idx, 0] - predicted_pts[start_idx:end_idx, 0]))},'
            f'{int(np.average(true_pts[start_idx:end_idx, 1] - predicted_pts[start_idx:end_idx, 1]))}, '
            f'{int(np.average(true_pts[start_idx:end_idx, 2] - predicted_pts[start_idx:end_idx, 2]))})')
        ax.set_xlabel('Index')
        ax.set_ylabel('Error [mm]')

    plt.tight_layout()
    plt.show()
    if filter_outlier:
        print(f"Number of outliers: {outlier_count}")
    fig.tight_layout()
    plt.show()


def RMSE_test(true_pts, predicted_pts, points_per_plot=121, mad_thresh=4):
    true_pts = np.array(true_pts)
    predicted_pts = np.array(predicted_pts)

    # # Calculate Median Absolute Deviation (MAD) for each dimension
    # mad_x = np.median(np.abs(predicted_pts[:, 0] - np.median(predicted_pts[:, 0])))
    # mad_y = np.median(np.abs(predicted_pts[:, 1] - np.median(predicted_pts[:, 1])))
    # mad_z = np.median(np.abs(predicted_pts[:, 2] - np.median(predicted_pts[:, 2])))
    #
    # # Set thresholds as 3 times MAD
    # thresh_x = mad_thresh * mad_x * 1.4826
    # thresh_y = mad_thresh * mad_y * 1.4826
    # thresh_z = mad_thresh * mad_z * 1.4826
    #
    # # Identify outliers based on MAD
    # outliers_x = np.abs(predicted_pts[:, 0] - np.median(predicted_pts[:, 0])) > thresh_x
    # outliers_y = np.abs(predicted_pts[:, 1] - np.median(predicted_pts[:, 1])) > thresh_y
    # outliers_z = np.abs(predicted_pts[:, 2] - np.median(predicted_pts[:, 2])) > thresh_z
    #
    # # Combine outliers across all dimensions
    # outliers = outliers_x | outliers_y | outliers_z
    #
    # # Remove outliers
    # true_pts_clean = true_pts[~outliers]
    # predicted_pts_clean = predicted_pts[~outliers]

    # Plot RMSE for each dimension
    fig, ax = plt.subplots(figsize=(8, 6))

    rmse_arr = []
    num_plot = (len(true_pts) // points_per_plot)
    for i in range(num_plot):
        start_idx = i * points_per_plot
        end_idx = (i + 1) * points_per_plot

        rmse_x = np.sqrt(
            np.mean((predicted_pts[start_idx:end_idx, 0] - true_pts[start_idx:end_idx, 0]) ** 2))
        rmse_y = np.sqrt(
            np.mean((predicted_pts[start_idx:end_idx, 1] - true_pts[start_idx:end_idx, 1]) ** 2))
        rmse_z = np.sqrt(
            np.mean((predicted_pts[start_idx:end_idx, 2] - true_pts[start_idx:end_idx, 2]) ** 2))

        rmse_arr.append([rmse_x, rmse_y, rmse_z])

        ax.plot(i, rmse_x, 'g^', label='RMSE X' if i == 0 else "")
        ax.plot(i, rmse_y, 'bo', label='RMSE Y' if i == 0 else "")
        ax.plot(i, rmse_z, 'rs', label='RMSE Z' if i == 0 else "")

    maximum_y = np.max(np.array(rmse_arr).flatten())
    std = np.std(np.array(rmse_arr).flatten())

    ax.set_xlabel('Plot Index')
    ax.set_ylabel('RMSE [mm]')
    ax.legend(loc="upper right")
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    plt.ylim(0, maximum_y + std)
    plt.show()


def MAE_test(true_pts, predicted_pts, points_per_plot=121, mad_thresh=4):
    true_pts = np.array(true_pts)
    predicted_pts = np.array(predicted_pts)

    # Plot MAE for each dimension
    fig, ax = plt.subplots(figsize=(8, 6))

    mae_arr = []
    num_plot = (len(true_pts) // points_per_plot)
    for i in range(num_plot):
        start_idx = i * points_per_plot
        end_idx = (i + 1) * points_per_plot

        mae_x = np.mean(np.abs(predicted_pts[start_idx:end_idx, 0] - true_pts[start_idx:end_idx, 0]))
        mae_y = np.mean(np.abs(predicted_pts[start_idx:end_idx, 1] - true_pts[start_idx:end_idx, 1]))
        mae_z = np.mean(np.abs(predicted_pts[start_idx:end_idx, 2] - true_pts[start_idx:end_idx, 2]))

        mae_arr.append([mae_x, mae_y, mae_z])

        ax.plot(i, mae_x, 'g^', label='MAE_X' if i == 0 else "")
        ax.plot(i, mae_y, 'bo', label='MAE_Y' if i == 0 else "")
        ax.plot(i, mae_z, 'rs', label='MAE_Z' if i == 0 else "")

    maximum_y = np.max(np.array(mae_arr).flatten())
    std = np.std(np.array(mae_arr).flatten())

    ax.set_xlabel('Plot Index')
    ax.set_ylabel('MAE [mm]')
    ax.legend(loc="upper right")
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    plt.ylim(0, maximum_y + std)
    plt.show()


path = '../3D_Experiment/'
left_coeff = np.load(os.path.join(path, "Calibration/left_cam_coeff.npy"))
right_coeff = np.load(os.path.join(path, "Calibration/right_cam_coeff.npy"))
inverse_object = ils.InverseTransform(left_coeff, right_coeff,
                                      i=3, j=3, k=2)

corner_detection_object = cd.CornerDetector(os.path.join(path, "Test"), 5, (40, 40))

# dx = [-3, -3, -3, 0, 2, 3]
# dy = [3, 3, 0, -1, 3, 3]

dx = [-3, 0, -3, 2]
dy = [0, -1, 3, 3]
x_offset = 100
y_offset = 100

path_test = os.path.join(path, "Test")
if not (os.path.exists(os.path.join(path_test, "left_camera_pos.npy")) and
        os.path.exists(os.path.join(path_test, "right_camera_pos.npy")) and
        os.path.exists(os.path.join(path_test, "3D_camera_pos.npy"))):

    true_pt = corner_detection_object.get_initial_calibrate_corners(x_offset=x_offset, y_offset=y_offset,
                                                                    dx=dx, dy=dy)
    initial_pt = corner_detection_object.get_initial_calibrate_corners()
    left_pt = corner_detection_object.get_left_corners()
    right_pt = corner_detection_object.get_right_corners()
else:
    initial_pt = np.load(os.path.join(path_test, "3D_camera_pos.npy"), allow_pickle=True)
    left_pt = np.load(os.path.join(path_test, "left_camera_pos.npy"), allow_pickle=True)
    right_pt = np.load(os.path.join(path_test, "right_camera_pos.npy"), allow_pickle=True)
    true_pt = np.load(os.path.join(path_test, "3D_true_camera_pos.npy"), allow_pickle=True)

predicted_3D_pos = []
print("## Calculating 3D Positions ##")
for point_left, point_right in zip(left_pt, right_pt):
    predicted_3D_pos.append(inverse_object.inverse_least_square(point_left, point_right))

print("## Plotting Visuals ##")
predicted_3D_pos = np.array(predicted_3D_pos)
MAE_test(true_pt, predicted_3D_pos, points_per_plot=36)
RMSE_test(true_pt, predicted_3D_pos, points_per_plot=36)
projection_error_display(np.array(true_pt), np.array(predicted_3D_pos), points_per_plot=36,
                         filter_outlier=False)
# projection_error_display(np.array(true_pt), np.array(predicted_3D_pos), points_per_plot=36)
