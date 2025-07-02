import os
import matplotlib.pyplot as plt
import numpy as np


def projection_object_to_image(params, point, num_params=19):
    """
    Function to project points on the object plane to the corresponding point on the
    image plane using the third order polynomial.
    :param num_params:      The number of parameters (coefficients)
    :param params:          The coefficients
    :param point:           The object plane points
    :return:                The transformed point on the image plane
    """
    params = params.reshape(num_params, 2)
    assert len(params) == num_params, "Parameters must contain 19 coefficient vectors."

    # Transformation Coefficient through Calibration
    coeff_x = params[:, 0]
    coeff_y = params[:, 1]
    assert len(coeff_x) and len(coeff_y) == num_params, "Coefficient must contain 19 values."

    xi, yi, zi = point

    # Define the polynomial model
    x_predicted = (coeff_x[0] +
                   (coeff_x[1] * xi) +
                   (coeff_x[2] * yi) +
                   (coeff_x[3] * zi) +
                   (coeff_x[4] * (xi ** 2)) +
                   (coeff_x[5] * xi * yi) +
                   (coeff_x[6] * (yi ** 2)) +
                   (coeff_x[7] * (xi * zi)) +
                   (coeff_x[8] * (yi * zi)) +
                   (coeff_x[9] * (zi ** 2)) +
                   (coeff_x[10] * (xi ** 3)) +
                   (coeff_x[11] * (xi ** 2) * yi) +
                   (coeff_x[12] * xi * (yi ** 2)) +
                   (coeff_x[13] * (yi ** 3)) +
                   (coeff_x[14] * (xi ** 2) * zi) +
                   (coeff_x[15] * xi * yi * zi) +
                   (coeff_x[16] * (yi ** 2) * zi) +
                   (coeff_x[17] * xi * (zi ** 2)) +
                   (coeff_x[18] * yi * (zi ** 2)))

    y_predicted = (coeff_y[0] +
                   (coeff_y[1] * xi) +
                   (coeff_y[2] * yi) +
                   (coeff_y[3] * zi) +
                   (coeff_y[4] * (xi ** 2)) +
                   (coeff_y[5] * xi * yi) +
                   (coeff_y[6] * (yi ** 2)) +
                   (coeff_y[7] * (xi * zi)) +
                   (coeff_y[8] * (yi * zi)) +
                   (coeff_y[9] * (zi ** 2)) +
                   (coeff_y[10] * (xi ** 3)) +
                   (coeff_y[11] * (xi ** 2) * yi) +
                   (coeff_y[12] * xi * (yi ** 2)) +
                   (coeff_y[13] * (yi ** 3)) +
                   (coeff_y[14] * (xi ** 2) * zi) +
                   (coeff_y[15] * xi * yi * zi) +
                   (coeff_y[16] * (yi ** 2) * zi) +
                   (coeff_y[17] * xi * (zi ** 2)) +
                   (coeff_y[18] * yi * (zi ** 2)))

    return x_predicted, y_predicted


def projection_error_display(calibrate_coeff, calibrated_pts, distorted_pts, name, points_per_plot=121):
    num_figure = (len(calibrated_pts) + points_per_plot - 1) // points_per_plot
    cols = (num_figure + 2) // 3

    fig, axes = plt.subplots(nrows=3, ncols=cols, figsize=(4 * cols, 6))
    axes = axes.flatten()

    all_errors = []
    for i in range(len(calibrated_pts)):
        u, v = projection_object_to_image(calibrate_coeff, calibrated_pts[i], num_params=19)
        predicted = np.array([u, v])
        error = np.sqrt(np.sum((distorted_pts[i] - predicted) ** 2))
        all_errors.append(error)

    z_coord = calibrated_pts[::points_per_plot, 2]
    all_error = np.array(all_errors).reshape((num_figure, points_per_plot))
    overall_max_error = np.max(all_errors) + np.std(all_errors)

    for i, ax in enumerate(axes):
        if i < num_figure:
            ax.scatter(np.arange(points_per_plot), all_error[i, :], c='k', label='Projection Error')
            ax.set_title(f'Calibration: z = {z_coord[i]} mm')
            ax.set_ylim(0, overall_max_error)
        else:
            ax.axis('off')

    fig.supylabel("Error [Pixel]")
    fig.supxlabel('Index')
    fig.suptitle(f"{name} Projection Error")
    fig.tight_layout()


def projection_rmse(calibrate_coeff_l, calibrate_coeff_r, calibrated_pts, left_pt_arr, right_pt_arr):
    rmse_l, rmse_r = [], []

    def rmse(pred, truth):
        u_diff = pred[:, 0] - truth[:, 0]
        v_diff = pred[:, 1] - truth[:, 1]

        u_rmse = np.sqrt(np.mean(np.square(u_diff), axis=0))
        v_rmse = np.sqrt(np.mean(np.square(v_diff), axis=0))

        return [u_rmse, v_rmse]

    for i in range(len(calibrated_pts)):
        predicted_l = np.array([projection_object_to_image(calibrate_coeff_l[i], pt, 19) for pt in calibrated_pts[i]])
        rmse_l.append(rmse(predicted_l, left_pt_arr[i]))

        predicted_r = np.array([projection_object_to_image(calibrate_coeff_r[i], pt, 19) for pt in calibrated_pts[i]])
        rmse_r.append(rmse(predicted_r, right_pt_arr[i]))

    x_labels = ["3", "5", "7", "9"]

    fig, ax = plt.subplots()
    x = range(len(rmse_l))

    left_u = [l[0] for l in rmse_l]
    left_v = [l[1] for l in rmse_l]
    right_u = [r[0] for r in rmse_r]
    right_v = [r[1] for r in rmse_r]

    ax.scatter(x, left_u, label='Left X', marker='o')
    ax.scatter(x, left_v, label='Left Y', marker='s')
    ax.scatter(x, right_u, label='Right X', marker='^')
    ax.scatter(x, right_v, label='Right Y', marker='*')

    ax.plot(x, left_u, 'o--', ms=2)
    ax.plot(x, left_v, 's--', ms=2)
    ax.plot(x, right_u, '^--', ms=2)
    ax.plot(x, right_v, '*--', ms=2)

    ax.set_xlabel('Number of Calibration Set')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('RMSE')
    ax.legend()


def mean_range(calibrate_coeff_l, calibrate_coeff_r, calibrated_pts, left_pt_arr, right_pt_arr):
    distance_l, distance_r = [], []
    for i in range(len(calibrated_pts)):
        predicted_l = np.array([projection_object_to_image(calibrate_coeff_l[i], pt, 19) for pt in calibrated_pts[i]])
        predicted_r = np.array([projection_object_to_image(calibrate_coeff_r[i], pt, 19) for pt in calibrated_pts[i]])

        distance_l.append(np.sqrt(np.sum((predicted_l - left_pt_arr[i]).astype(float) ** 2, axis=1)))
        distance_r.append(np.sqrt(np.sum((predicted_r - right_pt_arr[i]).astype(float) ** 2, axis=1)))

    data = []
    for set_l, set_r in zip(distance_l, distance_r):
        data.append(set_l)  # Append distances for the left camera for the current set
        data.append(set_r)  # Append distances for the right camera for the current set

    # Create the box plot
    fig, ax = plt.subplots()
    num_calibration_sets = len(calibrated_pts)
    positions = np.arange(1, num_calibration_sets * 2 * 2, 2)  # Space the box plots evenly

    ax.boxplot(data, positions=positions)

    # Set the x-axis labels
    x_labels = ["3", "5", "7", "9"]

    # Set the x-axis tick positions
    tick_positions = np.arange(1.5, num_calibration_sets * 2 * 2, 4)  # Center ticks between pairs of box plots
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='right')

    # Set the limits for the x-axis
    ax.set_xlim(0, positions[-1] + 1)

    # Display the plot
    fig.supylabel("Distance [Pixel]")
    fig.supxlabel("Number of Calibration Sets")
    plt.tight_layout()  # Adjust layout to make room for x-labels
    # plt.show()


left_coeff_arr, right_coeff_arr = [], []
calibrate_pt_arr, left_arr, right_arr = [], [], []
for i in range(4):
    set_path = f"../3D_Experiment/Calibration/Set_{i + 1}"
    left_coeff = np.load(os.path.join(set_path, "left_cam_coeff.npy"))
    right_coeff = np.load(os.path.join(set_path, "right_cam_coeff.npy"))

    left_coeff_arr.append(left_coeff)
    right_coeff_arr.append(right_coeff)

    calibrate_pt = np.load(os.path.join(set_path, "calibrate_camera_pt.npy"))
    left_pt = np.array(np.load(os.path.join(set_path, "left_camera_pt.npy"), allow_pickle=True))
    right_pt = np.array(np.load(os.path.join(set_path, "right_camera_pt.npy"), allow_pickle=True))

    calibrate_pt_arr.append(calibrate_pt)
    left_arr.append(left_pt)
    right_arr.append(right_pt)

    projection_error_display(left_coeff, calibrate_pt, left_pt, "Left")
    projection_error_display(right_coeff, calibrate_pt, right_pt, "Right")
mean_range(left_coeff_arr, right_coeff_arr, calibrate_pt_arr, left_arr, right_arr)
projection_rmse(left_coeff_arr, right_coeff_arr, calibrate_pt_arr, left_arr, right_arr)
plt.show()
