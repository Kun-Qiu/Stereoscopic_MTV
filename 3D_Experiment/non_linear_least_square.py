import numpy as np
from scipy.optimize import least_squares
import os


def projection_object_to_image(num_params, params, point):
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

    xi, yi, zi = np.intp(point)

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


class CalibrationTransformation:
    def __init__(self, name_camera, calibrated_points=None, distorted_points=None):
        assert len(calibrated_points) == len(distorted_points), "Length of calibration and distortion are not equal."
        assert name_camera.lower == "left" or "right"

        self.__calibrated_points = calibrated_points
        self.__distorted_points = distorted_points
        self.__NUM_PARAM = 19
        self.__left_calibrate_param = np.zeros((self.__NUM_PARAM, 2))
        self.__right_calibrate_param = np.zeros((self.__NUM_PARAM, 2))
        self.__name = name_camera

    def __calibrate_residuals(self, params):
        """
        The residual term of the nonlinear least square --> Predicted - Truth

        :param params:      The (19 x 2) parameter for the polynomial fitting
        :return:            The residual --> Predicted - Truth
        """
        x_predicted, y_predicted = [], []
        for point in self.__calibrated_points:
            x_img, y_img = self.__calibration_polynomial(point, params)
            x_predicted.append(x_img)
            y_predicted.append(y_img)

        x_predicted = np.array(x_predicted)
        y_predicted = np.array(y_predicted)

        x_true, y_true = zip(*self.__distorted_points)
        x_true = np.array(x_true)
        y_true = np.array(y_true)

        return np.concatenate([x_predicted - x_true, y_predicted - y_true]).flatten()

    def __calibration_polynomial(self, point, params):
        """
        Given the predicted coefficient, determine the transformation onto the image plane from an
        object plane

        :param point:       Point located in the object plane
        :param params:      Predicted coefficients used to project object plane onto image plane
        :return:            Image plane points of the inputted object plane points
        """
        params = params.reshape(self.__NUM_PARAM, 2)
        assert len(params) == 19, "Parameters must contain 19 coefficient vectors."

        # Transformation Coefficient through Calibration
        coeff_x = params[:, 0]
        coeff_y = params[:, 1]
        assert len(coeff_x) and len(coeff_y) == 19, "Coefficient must contain 19 values."

        xi, yi, zi = np.intp(point)

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

    def calibrate_least_square(self):
        """
        Apply the nonlinear least square using algorithm provided by SciPy Optimization Library

        :return: The optimized solution of the coefficient that minimize the residual
        """
        if self.__calibrated_points is None or self.__distorted_points is None:
            raise ValueError(f"This function cannot be called if no input corners are given")
        else:
            params = np.zeros((self.__NUM_PARAM, 2))
            result = least_squares(self.__calibrate_residuals, params.flatten(), method='trf')

            if self.__name == "left":
                self.__left_calibrate_param = result.x.reshape((self.__NUM_PARAM, 2))
            else:
                self.__right_calibrate_param = result.x.reshape((self.__NUM_PARAM, 2))

    def get_camera_transform_function(self):
        if self.__name == "left":
            return self.__left_calibrate_param
        else:
            return self.__right_calibrate_param

    def set_right_calibration(self, rhs):
        assert not np.all(rhs.get_right_calibration == 0), "The right hand side does not have calibration coefficients."
        self.__right_calibrate_param = rhs.get_right_calibration()

    def get_right_calibration(self):
        assert not np.all(self.__right_calibrate_param == 0), "The coefficient are all zeros."
        return self.__right_calibrate_param

    def set_left_calibration(self, lhs):
        assert not np.all(lhs.get_left_calibration == 0), "The left hand side does not have calibration coefficients."
        self.__left_calibrate_param = lhs.get_left_calibration()

    def get_left_calibration(self):
        assert not np.all(self.__left_calibrate_param == 0), "The coefficients are all zeros."
        return self.__left_calibrate_param

    def save_calibration_coefficient(self, path):
        assert not np.all(self.__left_calibrate_param == 0), "Left camera calibration coefficients are all zeros."
        assert not np.all(self.__left_calibrate_param == 0), "Right camera calibration coefficients are all zeros."

        np.save(os.path.join(path, "left_cam_coeff.npy"), self.__left_calibrate_param)
        np.save(os.path.join(path, "right_cam_coeff.npy"), self.__right_calibrate_param)
