import os, csv
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


def read_csv_points(csv_path):
    """
    Reads CSV with headers: Camera# X Y x y z
    """
    assert os.path.exists(csv_path), f"CSV path {csv_path} does not exist."
    
    points = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract columns as floats
            X = float(row['X'])
            Y = float(row['Y'])
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            points.append([X, Y, x, y, z])
    return np.array(points)


def plot_projection_error(coeff_path,  csv_path, plot=True):

    coeff = np.load(coeff_path, allow_pickle=True)
    points = read_csv_points(csv_path)

    # Extract calibration and actual points
    calibrate_pt = points[..., 2:5]   # object points (x, y, z)
    actual_image_pt = points[..., 0:2] # actual image points (X, Y) from CSV
    estimated_projection = np.array([projection_object_to_image(coeff, pt) for pt in calibrate_pt])

    # Compute absolute error with shape: (num_points, 2)
    abs_error = np.abs(estimated_projection - actual_image_pt)  
    abs_error_x = abs_error[:, 0]
    abs_error_y = abs_error[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plots
    sc1 = ax1.scatter(actual_image_pt[:, 0], actual_image_pt[:, 1], c=abs_error_x, cmap='Greys', marker='o')
    ax1.set_xlabel('X Coordinate (pixels)', fontsize=14)
    ax1.set_ylabel('Y Coordinate (pixels)', fontsize=14)
    ax1.set_title('X-Component Error', fontsize=14)
    ax1.invert_yaxis()

    sc2 = ax2.scatter(actual_image_pt[:, 0], actual_image_pt[:, 1], c=abs_error_y, cmap='Greys', marker='o')
    ax2.set_xlabel('X Coordinate (pixels)', fontsize=14)
    ax2.set_title('Y-Component Error', fontsize=14)
    ax2.invert_yaxis()

    # Add shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label('Error (pixels)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Adjust layout to fit colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    
    if plot:
        plt.show()


if __name__ == "__main__":
    left_coeff_path = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\experiment\left_cam_coeff.npy"
    left_csv_path = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\camera_points_filtered.csv"
    left_coeff = np.load(left_coeff_path, allow_pickle=True)

    right_csv_path = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\camera_points_2_filtered.csv"
    right_coeff_path = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\experiment\right_cam_coeff.npy"
    right_coeff = np.load(right_coeff_path, allow_pickle=True)

    plot_projection_error(left_coeff_path, left_csv_path, plot=False)
    plot_projection_error(right_coeff_path, right_csv_path, plot=True)

