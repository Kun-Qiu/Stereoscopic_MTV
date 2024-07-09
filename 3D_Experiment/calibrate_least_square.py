import numpy as np
from scipy.optimize import least_squares
import os


class CalibrationTransformation:
    def __init__(self, calibrated_points, distorted_points, num_square=10):
        assert len(calibrated_points) == len(distorted_points), "Length of calibration and distortion are not equal."

        self.__calibrated_points = calibrated_points
        self.__distorted_points = distorted_points
        self.__NUM_PARAM = 19
        self.__NUM_SQUARE = num_square
        self.__calibrate_param = np.zeros((self.__NUM_PARAM, 2))

    def __residuals(self, params):
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

        xi, yi, zi = point

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
            result = least_squares(self.__residuals, params.flatten(), method='trf',
                                   xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy')

            self.__calibrate_param = result.x.reshape((self.__NUM_PARAM, 2))

    def get_camera_transform_function(self):
        """
        :return: Camera transformation coefficients
        """
        return self.__calibrate_param

    def clear_calibration_param(self):
        """
        Clears the camera transformation coefficients
        :return: None
        """
        self.__calibrate_param = np.array((self.__NUM_PARAM, 2))

    def set_calibration_param(self, coeffs):
        """
        Set the object's transformation to a user defined coefficient

        :param coeffs:  User defined coefficient
        :return:        None
        """
        assert coeffs.shape == (self.__NUM_PARAM, 2), \
            f"Input parameter does not have the shape of ({self.__NUM_PARAM}, 2)."
        self.__calibrate_param = coeffs

    def save_calibration_coefficient(self, path, name):
        """
        Save the transformation coefficient to a desired path

        :param path:    Path of the destination
        :param name:    Name of the file
        :return:        None
        """
        assert not np.all(self.__calibrate_param == 0), "Camera calibration coefficients cannot be all zeros."

        np.save(os.path.join(path, name), self.__calibrate_param)
