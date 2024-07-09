import numpy as np
from scipy.optimize import least_squares


class InverseTransform:
    def __init__(self, left_coefficient, right_coefficient):
        self.__NUM_PARAM = 19

        assert left_coefficient.shape == right_coefficient.shape == (self.__NUM_PARAM, 2), \
            f"Shape of left camera or right camera coefficient is not ({self.__NUM_PARAM}, 2)."

        self.__left_calibrate_coeff = left_coefficient
        self.__right_calibrate_coeff = right_coefficient

    def __inverse_polynomial_transform(self, predicted_object_pt, img_pt_left, img_pt_right):
        """
        Residual for the prediction based on the inputted image point from the left and right camera

        :param predicted_object_pt:     Predicted object plane point (x, y, z)
        :param img_pt_left:             The corresponding image plane coordinate on the left camera
        :param img_pt_right:            The corresponding image plane coordinate on the right camera
        :return:                        The difference between predicted (X, Y) and actual (X, Y)
        """
        assert img_pt_left.shape == img_pt_right.shape == (2,), "Shape of input image coordinates must be (2,)."

        xi, yi, zi = predicted_object_pt

        coeff_x_left, coeff_y_left = self.__left_calibrate_coeff[:, 0], self.__left_calibrate_coeff[:, 1]
        coeff_x_right, coeff_y_right = self.__right_calibrate_coeff[:, 0], self.__right_calibrate_coeff[:, 1]

        eq1 = (coeff_x_left[0] + coeff_x_left[1] * xi + coeff_x_left[2] * yi +
               coeff_x_left[3] * zi + coeff_x_left[4] * (xi ** 2) + coeff_x_left[5] * xi * yi +
               coeff_x_left[6] * (yi ** 2) + coeff_x_left[7] * (xi * zi) + coeff_x_left[8] * (yi * zi) +
               coeff_x_left[9] * (zi ** 2) + coeff_x_left[10] * (xi ** 3) +
               coeff_x_left[11] * (xi ** 2) * yi + coeff_x_left[12] * xi * (yi ** 2) +
               coeff_x_left[13] * (yi ** 3) + coeff_x_left[14] * (xi ** 2) * zi +
               coeff_x_left[15] * xi * yi * zi + coeff_x_left[16] * (yi ** 2) * zi +
               coeff_x_left[17] * xi * (zi ** 2) + coeff_x_left[18] * yi * (zi ** 2)) - img_pt_left[0]

        eq2 = (coeff_y_left[0] + coeff_y_left[1] * xi + coeff_y_left[2] * yi +
               coeff_y_left[3] * zi + coeff_y_left[4] * (xi ** 2) + coeff_y_left[5] * xi * yi +
               coeff_y_left[6] * (yi ** 2) + coeff_y_left[7] * (xi * zi) + coeff_y_left[8] * (yi * zi) +
               coeff_y_left[9] * (zi ** 2) + coeff_y_left[10] * (xi ** 3) +
               coeff_y_left[11] * (xi ** 2) * yi + coeff_y_left[12] * xi * (yi ** 2) +
               coeff_y_left[13] * (yi ** 3) + coeff_y_left[14] * (xi ** 2) * zi +
               coeff_y_left[15] * xi * yi * zi + coeff_y_left[16] * (yi ** 2) * zi +
               coeff_y_left[17] * xi * (zi ** 2) + coeff_y_left[18] * yi * (zi ** 2)) - img_pt_left[1]

        eq3 = (coeff_x_right[0] + coeff_x_right[1] * xi + coeff_x_right[2] * yi +
               coeff_x_right[3] * zi + coeff_x_right[4] * (xi ** 2) + coeff_x_right[5] * xi * yi +
               coeff_x_right[6] * (yi ** 2) + coeff_x_right[7] * (xi * zi) + coeff_x_right[8] * (yi * zi) +
               coeff_x_right[9] * (zi ** 2) + coeff_x_right[10] * (xi ** 3) +
               coeff_x_right[11] * (xi ** 2) * yi + coeff_x_right[12] * xi * (yi ** 2) +
               coeff_x_right[13] * (yi ** 3) + coeff_x_right[14] * (xi ** 2) * zi +
               coeff_x_right[15] * xi * yi * zi + coeff_x_right[16] * (yi ** 2) * zi +
               coeff_x_right[17] * xi * (zi ** 2) + coeff_x_right[18] * yi * (zi ** 2)) - img_pt_right[0]

        eq4 = (coeff_y_right[0] + coeff_y_right[1] * xi + coeff_y_right[2] * yi +
               coeff_y_right[3] * zi + coeff_y_right[4] * (xi ** 2) + coeff_y_right[5] * xi * yi +
               coeff_y_right[6] * (yi ** 2) + coeff_y_right[7] * (xi * zi) + coeff_y_right[8] * (yi * zi) +
               coeff_y_right[9] * (zi ** 2) + coeff_y_right[10] * (xi ** 3) +
               coeff_y_right[11] * (xi ** 2) * yi + coeff_y_right[12] * xi * (yi ** 2) +
               coeff_y_right[13] * (yi ** 3) + coeff_y_right[14] * (xi ** 2) * zi +
               coeff_y_right[15] * xi * yi * zi + coeff_y_right[16] * (yi ** 2) * zi +
               coeff_y_right[17] * xi * (zi ** 2) + coeff_y_right[18] * yi * (zi ** 2)) - img_pt_right[1]

        return [eq1, eq2, eq3, eq4]

    def inverse_least_square(self, left_img_pts, right_img_pts):
        """
        Apply the nonlinear least square using algorithm provided by SciPy Optimization Library

        :param left_img_pts:        Image plane point from the left camera
        :param right_img_pts:       Image plane point from the right camera
        :return:                    Optimized (x, y, z)
        """
        object_pt_predicted = np.array(((left_img_pts[0] + right_img_pts[0]) / 2,
                                        (left_img_pts[1] + right_img_pts[1]) / 2,
                                        0), dtype=np.float64)

        result = least_squares(self.__inverse_polynomial_transform, x0=object_pt_predicted, method='trf',
                               xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy',
                               args=(left_img_pts, right_img_pts))

        return result.x
