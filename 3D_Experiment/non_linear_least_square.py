import numpy as np
from scipy.optimize import least_squares


def projection_object_to_image(num_params, params, point):
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


class NonLinearLeastSquare:
    def __init__(self, calibrated_points=None, distorted_points=None):
        assert len(calibrated_points) == len(distorted_points), "Length of calibration and distortion are not equal."
        self.__calibrated_points = calibrated_points
        self.__distorted_points = distorted_points
        self.__NUM_PARAM = 19
        self.__left_calibrate_param = np.zeros(self.__NUM_PARAM, 2)
        self.__right_calibrate_param = np.zeros(self.__NUM_PARAM, 2)

    def __calibrate_residuals(self, params):
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
        params = params.reshape(self.__NUM_PARAM, 2)
        assert len(params) == 19, "Parameters must contain 19 coefficient vectors."

        # Transformation Coefficient through Calibration
        coeff_x = params[:, 0]
        coeff_y = params[:, 1]
        assert len(coeff_x) and len(coeff_y) == 19, "Coefficient must contain 19 values."

        xi, yi, zi = np.intp(point)

        # Polynomial model
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

    def __inverse_residuals(self, object_pt, img_pt_left, img_pt_right, coeffs_left, coeffs_right):
        assert img_pt_left.shape == img_pt_right.shape == (2,), "Shape of input image coordinates must be (2,)."

        assert coeffs_left.shape == coeffs_right.shape == (self.__NUM_PARAM, 2), \
            "Coefficients must be of shape (19, 2)."

        xi, yi, zi = np.intp(object_pt)

        coeff_x_left, coeff_y_left = coeffs_left[:, 0], coeffs_left[:, 1]
        coeff_x_right, coeff_y_right = coeffs_right[:, 0], coeffs_right[:, 1]

        x_predicted_left = (coeff_x_left[0] + coeff_x_left[1] * xi + coeff_x_left[2] * yi +
                            coeff_x_left[3] * zi + coeff_x_left[4] * (xi ** 2) + coeff_x_left[5] * xi * yi +
                            coeff_x_left[6] * (yi ** 2) + coeff_x_left[7] * (xi * zi) + coeff_x_left[8] * (yi * zi) +
                            coeff_x_left[9] * (zi ** 2) + coeff_x_left[10] * (xi ** 3) +
                            coeff_x_left[11] * (xi ** 2) * yi + coeff_x_left[12] * xi * (yi ** 2) +
                            coeff_x_left[13] * (yi ** 3) + coeff_x_left[14] * (xi ** 2) * zi +
                            coeff_x_left[15] * xi * yi * zi + coeff_x_left[16] * (yi ** 2) * zi +
                            coeff_x_left[17] * xi * (zi ** 2) + coeff_x_left[18] * yi * (zi ** 2))

        y_predicted_left = (coeff_y_left[0] + coeff_y_left[1] * xi + coeff_y_left[2] * yi +
                            coeff_y_left[3] * zi + coeff_y_left[4] * (xi ** 2) + coeff_y_left[5] * xi * yi +
                            coeff_y_left[6] * (yi ** 2) + coeff_y_left[7] * (xi * zi) + coeff_y_left[8] * (yi * zi) +
                            coeff_y_left[9] * (zi ** 2) + coeff_y_left[10] * (xi ** 3) +
                            coeff_y_left[11] * (xi ** 2) * yi + coeff_y_left[12] * xi * (yi ** 2) +
                            coeff_y_left[13] * (yi ** 3) + coeff_y_left[14] * (xi ** 2) * zi +
                            coeff_y_left[15] * xi * yi * zi + coeff_y_left[16] * (yi ** 2) * zi +
                            coeff_y_left[17] * xi * (zi ** 2) + coeff_y_left[18] * yi * (zi ** 2))

        x_predicted_right = (coeff_x_right[0] + coeff_x_right[1] * xi + coeff_x_right[2] * yi +
                             coeff_x_right[3] * zi + coeff_x_right[4] * (xi ** 2) + coeff_x_right[5] * xi * yi +
                             coeff_x_right[6] * (yi ** 2) + coeff_x_right[7] * (xi * zi) + coeff_x_right[8] * (
                                     yi * zi) +
                             coeff_x_right[9] * (zi ** 2) + coeff_x_right[10] * (xi ** 3) +
                             coeff_x_right[11] * (xi ** 2) * yi + coeff_x_right[12] * xi * (yi ** 2) +
                             coeff_x_right[13] * (yi ** 3) + coeff_x_right[14] * (xi ** 2) * zi +
                             coeff_x_right[15] * xi * yi * zi + coeff_x_right[16] * (yi ** 2) * zi +
                             coeff_x_right[17] * xi * (zi ** 2) + coeff_x_right[18] * yi * (zi ** 2))

        y_predicted_right = (coeff_y_right[0] + coeff_y_right[1] * xi + coeff_y_right[2] * yi +
                             coeff_y_right[3] * zi + coeff_y_right[4] * (xi ** 2) + coeff_y_right[5] * xi * yi +
                             coeff_y_right[6] * (yi ** 2) + coeff_y_right[7] * (xi * zi) + coeff_y_right[8] * (
                                     yi * zi) +
                             coeff_y_right[9] * (zi ** 2) + coeff_y_right[10] * (xi ** 3) +
                             coeff_y_right[11] * (xi ** 2) * yi + coeff_y_right[12] * xi * (yi ** 2) +
                             coeff_y_right[13] * (yi ** 3) + coeff_y_right[14] * (xi ** 2) * zi +
                             coeff_y_right[15] * xi * yi * zi + coeff_y_right[16] * (yi ** 2) * zi +
                             coeff_y_right[17] * xi * (zi ** 2) + coeff_y_right[18] * yi * (zi ** 2))

        residuals = [x_predicted_left - img_pt_left[0], y_predicted_left - img_pt_left[1],
                     x_predicted_right - img_pt_right[0], y_predicted_right - img_pt_right[1]]
        return residuals

    def inverse_least_square(self, left_img_pts, right_img_pts, coeffs_left, coeffs_right):
        object_pt_predicated = np.zeros(3)
        result = least_squares(self.__inverse_residuals, object_pt_predicated, method='trf',
                               args=(left_img_pts, right_img_pts, coeffs_left, coeffs_right))
        return result.x

    def calibrate_least_square(self):
        if self.__calibrated_points is None or self.__distorted_points is None:
            raise ValueError(f"This function cannot be called if no input corners are given")
        else:
            params = np.zeros((self.__NUM_PARAM, 2))
            result = least_squares(self.__calibrate_residuals, params.flatten(), method='trf')
            return result.x.reshape((self.__NUM_PARAM, 2))

    def get_right_calibration(self, right_object):
        right_object.get_right_calibration
