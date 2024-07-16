import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

    if num_params == 20:
        x_predicted += (coeff_x[19] * (zi ** 3))
        y_predicted += (coeff_y[19] * (zi ** 3))

    return x_predicted, y_predicted


def projection_error_display(calibrate_coeff, calibrated_pts, distorted_pts, points_per_plot=121, num_param=19):
    num_points = len(calibrated_pts)

    fig = plt.figure(figsize=(8, 5))
    spec = gridspec.GridSpec(ncols=6, nrows=2)  # Define the grid layout similar to your provided code

    ax1 = fig.add_subplot(spec[0, 0:2])  # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[0, 2:4])
    ax3 = fig.add_subplot(spec[0, 4:])
    ax4 = fig.add_subplot(spec[1, 1:3])  # row 0 with axes spanning 2 cols on odds
    ax5 = fig.add_subplot(spec[1, 3:5])

    all_errors = []
    for i in range(num_points):
        u, v = projection_object_to_image(calibrate_coeff, calibrated_pts[i],
                                          num_params=num_param)
        predicted = np.array([u, v])
        error = np.sqrt(np.sum((distorted_pts[i] - predicted) ** 2))
        all_errors.append(error)

    overall_max_error = np.max(all_errors) + np.std(all_errors)

    for i in range(num_points):
        u, v = projection_object_to_image(calibrate_coeff, calibrated_pts[i],
                                          num_params=num_param)
        predicted = np.array([u, v])
        error = np.sqrt(np.sum((distorted_pts[i] - predicted) ** 2))
        z_coord = calibrated_pts[i, 2]

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
        else:
            continue

        ax.scatter(i % points_per_plot, error, c='k',
                   label='Projection Error' if (i % points_per_plot) == 0 else "")
        ax.set_title(f'Calibration: z = {z_coord} mm')
        ax.set_xlabel('Index')
        ax.set_ylabel('Error [Pixel]')
        ax.set_ylim(0, overall_max_error)  # Set the y-axis limits

    fig.tight_layout()
    plt.show()


path = '../3D_Experiment/'
left_coeff = np.load(os.path.join(path, "Calibration/left_cam_coeff.npy"))
right_coeff = np.load(os.path.join(path, "Calibration/right_cam_coeff.npy"))

calibrate_pt = np.load(os.path.join(path, "Calibration/calibrate_camera_pt.npy"))
left_pt = np.array(np.load(os.path.join(path, "Calibration/left_camera_pt.npy"), allow_pickle=True))
right_pt = np.array(np.load(os.path.join(path, "Calibration/right_camera_pt.npy"), allow_pickle=True))

projection_error_display(left_coeff, calibrate_pt, left_pt, num_param=19)
projection_error_display(right_coeff, calibrate_pt, right_pt, num_param=19)
