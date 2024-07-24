import os

import numpy as np
from scipy.optimize import least_squares


class CalibrationTransformation:
    def __init__(self, calibrated_points, distorted_points, num_square=10):
        assert len(calibrated_points) == len(distorted_points), "Length of calibration and distortion are not equal."

        self.__calibrated_points = np.array(calibrated_points).astype(float)
        self.__distorted_points = np.array(distorted_points).astype(float)
        self.__NUM_PARAM = 19
        self.__NUM_SQUARE = num_square
        self.__calibrate_param = np.zeros((self.__NUM_PARAM, 2))

    def __calibration_residuals_dx(self, coeff):
        """
        The residual term of the calibration procedure for projection onto the x coordinate of the
        image plane --> Predicted - Truth

        :param coeff    :   The (19 x 1) parameter for the polynomial fitting
        :return         :   The residual --> Predicted - Truth
        """

        return self.__soloff_polynomial(self.__calibrated_points, coeff) - self.__distorted_points[:, 0]

    def __calibration_residuals_dy(self, coeff):
        """
        The residual term of the calibration procedure for projection onto the y coordinate of the
        image plane --> Predicted - Truth

        :param coeff    :   The (19 x 1) parameter for the polynomial fitting
        :return         :   The residual --> Predicted - Truth
        """
        return self.__soloff_polynomial(self.__calibrated_points, coeff) - self.__distorted_points[:, 1]

    def __soloff_polynomial(self, XYZ, coeff):
        """
        Given the predicted coefficient, determine the transformation onto the image plane from an
        object plane

        :param XYZ      :   Points located in the object plane (calibration points)
        :param coeff    :   Predicted coefficients used to project object plane onto image plane
        :return         :   Image plane points of the inputted object plane points
        """
        assert len(coeff) == self.__NUM_PARAM, "Parameters must contain 19 coefficients."

        xi, yi, zi = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

        return (coeff[0] +
                (coeff[1] * xi) + (coeff[2] * yi) + (coeff[3] * zi) +
                (coeff[4] * (xi ** 2)) + (coeff[5] * xi * yi) +
                (coeff[6] * (yi ** 2)) + (coeff[7] * (xi * zi)) +
                (coeff[8] * (yi * zi)) + (coeff[9] * (zi ** 2)) +
                (coeff[10] * (xi ** 3)) + (coeff[11] * (xi ** 2) * yi) +
                (coeff[12] * xi * (yi ** 2)) + (coeff[13] * (yi ** 3)) +
                (coeff[14] * (xi ** 2) * zi) + (coeff[15] * xi * yi * zi) +
                (coeff[16] * (yi ** 2) * zi) + (coeff[17] * xi * (zi ** 2)) +
                (coeff[18] * yi * (zi ** 2)))

    def calibrate_least_square(self):
        """
        Apply the nonlinear least square to determine the transformation coefficients
        for the Soloff Polynomial.

        :return : The optimized solution of the coefficient that minimize the residual
        """
        if self.__calibrated_points is None or self.__distorted_points is None:
            raise ValueError(f"This function cannot be called if no input corners are given")
        else:
            params = np.zeros((10, 2))
            s_x = least_squares(self.__calibration_residuals_dx, params[:, 0], method='trf',
                                xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy').x
            s_y = least_squares(self.__calibration_residuals_dy, params[:, 1], method='trf',
                                xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy').x
            self.__calibrate_param = np.column_stack((s_x, s_y))

    def get_camera_transform_function(self):
        """
        Returns the transformation coefficient

        :return : Camera transformation coefficients
        """
        return self.__calibrate_param

    def clear_calibration_param(self):
        """
        Clears the camera transformation coefficients

        :return : None
        """
        self.__calibrate_param = np.array((self.__NUM_PARAM, 2))

    def set_calibration_param(self, coeffs):
        """
        Set the object's transformation to a user defined coefficient

        :param coeffs   :   User defined coefficient
        :return         :   None
        """
        assert coeffs.shape == (self.__NUM_PARAM, 2), \
            f"Input parameter does not have the shape of ({self.__NUM_PARAM}, 2)."
        self.__calibrate_param = coeffs

    def save_calibration_coefficient(self, path, name):
        """
        Save the transformation coefficient to a desired path

        :param path :   Path of the destination
        :param name :   Name of the file
        :return     :   Save the calibration coefficient to path
        """
        assert not np.all(self.__calibrate_param == 0), "Camera calibration coefficients cannot be all zeros."

        np.save(os.path.join(path, name), self.__calibrate_param)
