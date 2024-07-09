import os
import numpy as np
import inverse_least_square as ils
import matplotlib.pyplot as plt


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
    assert len(params) == 19, "Parameters must contain 19 coefficient vectors."

    # Transformation Coefficient through Calibration
    coeff_x = params[:, 0]
    coeff_y = params[:, 1]
    assert len(coeff_x) and len(coeff_y) == 19, "Coefficient must contain 19 values."

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


def projection_error_display(calibrate_coeff, calibrated_pts, distorted_pts, points_per_plot=121):
    num_points = len(calibrated_pts)
    num_plots = (num_points + points_per_plot - 1) // points_per_plot

    fig_error, axes_error = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))

    if num_plots == 1:
        axes_error = [axes_error]

    for i in range(num_points):
        u, v = projection_object_to_image(calibrate_coeff, calibrated_pts[i])
        predicted = np.array([u, v])
        error = np.sqrt(np.sum((distorted_pts[i] - predicted) ** 2))

        plot_index = i // points_per_plot
        axes_error[plot_index].scatter(i % points_per_plot, error, c='r',
                                       label='Projection Error' if (i % points_per_plot) == 0 else "")

    for ax in axes_error:
        ax.grid(False)

    fig_error.supylabel('Projection Error')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


path = '../3D_Experiment/'
left_coeff = np.load(os.path.join(path, "Calibration/left_cam_coeff.npy"))
right_coeff = np.load(os.path.join(path, "Calibration/right_cam_coeff.npy"))
calibrate_pt = np.load(os.path.join(path, "Calibration/calibrate_camera_pt.npy"))
left_pt = np.array(np.load(os.path.join(path, "Calibration/left_camera_pt.npy"), allow_pickle=True))
right_pt = np.array(np.load(os.path.join(path, "Calibration/right_camera_pt.npy"), allow_pickle=True))

# projection_error_display(left_coeff, calibrate_pt, left_pt)
# projection_error_display(right_coeff, calibrate_pt, right_pt)

# image_filenames = []
# if not os.path.isdir(os.path.join(path, "Test")):
#     print(f"Directory {os.path.join(path, 'Test')} does not exist.")
# else:
#     # Walk through the directory
#     for root, _, files in os.walk(os.path.join(path, "Test")):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#                 image_filenames.append(os.path.join(root, file))
#
# left_img = np.array(image_filenames[0:2])
# right_img = np.array(image_filenames[2:4])
#
left_corner = np.load(os.path.join(path, "Test/left_corner.npy"), allow_pickle=True)
right_corner = np.load(os.path.join(path, "Test/right_corner.npy"), allow_pickle=True)

img_000_left = left_corner[0:121]
img_000_right = right_corner[0:121]

img_221_left = left_corner[121:242]
img_221_right = right_corner[121:242]

inverse_object = ils.InverseTransform(left_coeff, right_coeff)

img_000_3D = []
for point_left, point_right in zip(img_221_left, img_221_right):
    img_000_3D.append(inverse_object.inverse_least_square(point_left, point_right))

print(img_000_3D)
7