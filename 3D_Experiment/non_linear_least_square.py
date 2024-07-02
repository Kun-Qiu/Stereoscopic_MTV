import numpy as np
from scipy.optimize import least_squares


class NonLinearLeastSquare:
    def __init__(self, calibrated_points, distorted_points):
        self._calibrated_points = calibrated_points
        self._distorted_points = distorted_points

    def _projection_polynomial_model(self, point, params):
        a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26, a31, a32, a34, a35, a36 = params
        a33 = 1

        xi, yi = np.intp(point)

        denominator = (a31 * xi) + (a32 * yi) + a33 + (a34 * xi ** 2) + \
                      (a35 * yi ** 2) + (a36 * xi * yi)

        x_predicted = ((a11 * xi) + (a12 * yi) + a13 + (a14 * xi ** 2) + (a15 * yi ** 2) +
                       (a16 * xi * yi)) / denominator

        y_predicted = ((a21 * xi) + (a22 * yi) + a23 + (a24 * xi ** 2) + (a25 * yi ** 2) +
                       (a26 * xi * yi)) / denominator
        return x_predicted, y_predicted

    def _residuals(self, params):
        x_predicted, y_predicted = [], []
        for point in self._distorted_points:
            x, y = self._projection_polynomial_model(point, params)
            x_predicted.append(x)
            y_predicted.append(y)
        x_predicted = np.array(x_predicted)
        y_predicted = np.array(y_predicted)

        x_true, y_true = zip(*self._calibrated_points)
        x_true = np.array(x_true)
        y_true = np.array(y_true)

        return np.concatenate([x_predicted - x_true, y_predicted - y_true])

    def calculate_least_square(self):
        params = np.random.rand(17)
        result = least_squares(self._residuals, params, method='lm')
        return result.x
