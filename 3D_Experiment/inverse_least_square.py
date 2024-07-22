import numpy as np
from scipy.optimize import least_squares


class InverseTransform:
    def __init__(self, left_coefficient, right_coefficient):
        self.__NUM_PARAM = 19
        assert left_coefficient.shape == right_coefficient.shape == (self.__NUM_PARAM, 2), \
            f"Shape of left camera or right camera coefficient is not ({self.__NUM_PARAM}, 2)."

        self.__left_calibrate_coeff = left_coefficient
        self.__right_calibrate_coeff = right_coefficient

    def __inverse_polynomial_transform_point(self, predicted_object_pt, img_pt_left, img_pt_right):
        """
        Residual for the prediction based on the inputted image point from the left and right camera

        :param predicted_object_pt  :   Predicted object plane point (x, y, z)
        :param img_pt_left          :   The corresponding image plane coordinate on the left camera
        :param img_pt_right         :   The corresponding image plane coordinate on the right camera
        :return                     :   The difference between predicted (X, Y) and actual (X, Y)
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
               coeff_x_left[17] * xi * (zi ** 2) + coeff_x_left[18] * yi * (zi ** 2))

        eq2 = (coeff_y_left[0] + coeff_y_left[1] * xi + coeff_y_left[2] * yi +
               coeff_y_left[3] * zi + coeff_y_left[4] * (xi ** 2) + coeff_y_left[5] * xi * yi +
               coeff_y_left[6] * (yi ** 2) + coeff_y_left[7] * (xi * zi) + coeff_y_left[8] * (yi * zi) +
               coeff_y_left[9] * (zi ** 2) + coeff_y_left[10] * (xi ** 3) +
               coeff_y_left[11] * (xi ** 2) * yi + coeff_y_left[12] * xi * (yi ** 2) +
               coeff_y_left[13] * (yi ** 3) + coeff_y_left[14] * (xi ** 2) * zi +
               coeff_y_left[15] * xi * yi * zi + coeff_y_left[16] * (yi ** 2) * zi +
               coeff_y_left[17] * xi * (zi ** 2) + coeff_y_left[18] * yi * (zi ** 2))

        eq3 = (coeff_x_right[0] + coeff_x_right[1] * xi + coeff_x_right[2] * yi +
               coeff_x_right[3] * zi + coeff_x_right[4] * (xi ** 2) + coeff_x_right[5] * xi * yi +
               coeff_x_right[6] * (yi ** 2) + coeff_x_right[7] * (xi * zi) + coeff_x_right[8] * (yi * zi) +
               coeff_x_right[9] * (zi ** 2) + coeff_x_right[10] * (xi ** 3) +
               coeff_x_right[11] * (xi ** 2) * yi + coeff_x_right[12] * xi * (yi ** 2) +
               coeff_x_right[13] * (yi ** 3) + coeff_x_right[14] * (xi ** 2) * zi +
               coeff_x_right[15] * xi * yi * zi + coeff_x_right[16] * (yi ** 2) * zi +
               coeff_x_right[17] * xi * (zi ** 2) + coeff_x_right[18] * yi * (zi ** 2))

        eq4 = (coeff_y_right[0] + coeff_y_right[1] * xi + coeff_y_right[2] * yi +
               coeff_y_right[3] * zi + coeff_y_right[4] * (xi ** 2) + coeff_y_right[5] * xi * yi +
               coeff_y_right[6] * (yi ** 2) + coeff_y_right[7] * (xi * zi) + coeff_y_right[8] * (yi * zi) +
               coeff_y_right[9] * (zi ** 2) + coeff_y_right[10] * (xi ** 3) +
               coeff_y_right[11] * (xi ** 2) * yi + coeff_y_right[12] * xi * (yi ** 2) +
               coeff_y_right[13] * (yi ** 3) + coeff_y_right[14] * (xi ** 2) * zi +
               coeff_y_right[15] * xi * yi * zi + coeff_y_right[16] * (yi ** 2) * zi +
               coeff_y_right[17] * xi * (zi ** 2) + coeff_y_right[18] * yi * (zi ** 2))

        return [eq1 - img_pt_left[0], eq2 - img_pt_left[1], eq3 - img_pt_right[0], eq4 - img_pt_right[1]]

    def __dFdx(self, XYZ, a):
        """
        Derivative of the Soloff polynomial in the x-direction

        :param XYZ  :   Input coordinate (xi, yi, zi)
        :param a    :   Transformation coefficient from calibration
        :return     :   Derivative in x direction
        """
        xi, yi, zi = XYZ

        assert a.shape == (self.__NUM_PARAM,), f"Coefficient has incorrect shape: {a.shape}."

        return (a[1] + 2 * a[4] * xi + a[5] * yi + a[7] * zi +
                3 * a[10] * pow(xi, 2) + 2 * a[11] * xi * yi +
                a[12] * pow(yi, 2) + 2 * a[14] * xi * zi +
                a[15] * yi * zi + a[17] * pow(zi, 2))

    def __dFdy(self, XYZ, a):
        """
        Derivative of the Soloff polynomial in the y-direction

        :param XYZ  :   Input coordinate (xi, yi, zi)
        :param a    :   Transformation coefficient from calibration
        :return     :   Derivative in y direction
        """
        xi, yi, zi = XYZ

        assert a.shape == (self.__NUM_PARAM,), f"Coefficient has incorrect shape: {a.shape}."

        return (a[2] + a[5] * xi + 2 * a[6] * yi + a[8] * zi +
                a[11] * pow(xi, 2) + 2 * a[12] * xi * yi +
                3 * a[13] * pow(yi, 2) + a[15] * xi * zi +
                2 * a[16] * yi * zi + a[18] * pow(zi, 2))

    def __dFdz(self, XYZ, a):
        """
        Derivative of the Soloff polynomial in the z-direction

        :param XYZ  :   Input coordinate (xi, yi, zi)
        :param a    :   Transformation coefficient from calibration
        :return     :   Derivative in z direction
        """
        xi, yi, zi = XYZ

        assert a.shape == (self.__NUM_PARAM,), f"Coefficient has incorrect shape: {a.shape}."

        return (a[3] + a[7] * xi + a[8] * yi + 2 * a[9] * zi +
                a[14] * pow(xi, 2) + a[15] * xi * yi + a[16] * pow(yi, 2) +
                2 * a[17] * xi * zi + 2 * a[18] * yi * zi)

    def __inverse_augmented_matrix(self, XYZ, a):
        """
        Obtain the augmented matrix for the transformation from object plane displacement to
        image plane displacements

        :param XYZ      :   Desired object plane position at (xi, yi, zi)
        :param a        :   The transformation coefficient from Soloff polynomial
        :return         :   The augmented displacement transformation matrix at the point
        """
        assert a.shape == (self.__NUM_PARAM, 2), f"Shape of coefficient must be {(self.__NUM_PARAM, 2)}. The input" \
                                                 f"coefficient have a shape of {a.shape}."

        coeff_x, coeff_y = a[:, 0], a[:, 1]

        # Left camera augmented matrix
        F11 = self.__dFdx(XYZ, coeff_x)
        F12 = self.__dFdy(XYZ, coeff_x)
        F13 = self.__dFdz(XYZ, coeff_x)

        F21 = self.__dFdx(XYZ, coeff_y)
        F22 = self.__dFdy(XYZ, coeff_y)
        F23 = self.__dFdz(XYZ, coeff_y)

        return F11, F12, F13, F21, F22, F23

    def __inverse_displacement_transform(self, dxyz, xyz, dXY_l, dXY_r):
        """
        Compute the residual between the predicted object plane displacement and the
        image plane displacements.

        :param dxyz     :   Predicted object plane displacement at point xyz (shape=(3,1))
        :param xyz      :   The point where displacement is desired (xi, yi, zi)
        :param dXY_l    :   Displacement of the point xyz in the left camera (shape=(2,1))
        :param dXY_r    :   Displacement of the point xyz in the right camera (shape=(2,1))
        :return         :   The residual
        """

        assert dxyz.shape == xyz.shape == (3,), \
            f"The shape of the input displacement and coordinate should be (3, 1). " \
            f"Shape of displacement: {dxyz.shape} and coordinate: {xyz.shape}."
        assert dXY_l.shape == dXY_r.shape == (2,), f"The shape of the camera displacements should be (3, 1). " \
                                                   f"Shape of left cam: {dXY_l.shape} and right cam: {dXY_r.shape}."

        dx, dy, dz = dxyz

        F11_1, F12_1, F13_1, F21_1, F22_1, F23_1 = self.__inverse_augmented_matrix(xyz, self.__left_calibrate_coeff)
        F11_2, F12_2, F13_2, F21_2, F22_2, F23_2 = self.__inverse_augmented_matrix(xyz, self.__right_calibrate_coeff)

        dX_l = F11_1 * dx + F12_1 * dy + F13_1 * dz
        dY_l = F21_1 * dx + F22_1 * dy + F23_1 * dz

        dX_r = F11_2 * dx + F12_2 * dy + F13_2 * dz
        dY_r = F21_2 * dx + F22_2 * dy + F23_2 * dz

        return [dXY_l[0] - dX_l, dXY_l[1] - dY_l,
                dXY_r[0] - dX_r, dXY_r[1] - dY_r]

    def inverse_point_least_square(self, left_img_pts, right_img_pts):
        """
        Apply the nonlinear least square using algorithm provided by SciPy Optimization Library

        ***
        Transform image plane coordinates in the object plane coordinate using the calibration
        coefficient.
        ***

        :param left_img_pts:        Image plane point from the left camera
        :param right_img_pts:       Image plane point from the right camera
        :return:                    Optimized (x, y, z)
        """
        object_pt_predicted = np.array(((left_img_pts[0] + right_img_pts[0]) / 2,
                                        (left_img_pts[1] + right_img_pts[1]) / 2,
                                        0), dtype=np.float64)

        result = least_squares(self.__inverse_polynomial_transform_point, x0=object_pt_predicted, method='trf',
                               xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy',
                               args=(left_img_pts, right_img_pts))

        return result.x

    def inverse_displacement(self, xyz, dXY_l, dXY_r):
        """
        Apply the nonlinear least square using algorithm provided by SciPy Optimization Library

        ***
        The function optimize for the object plane displacement based on the transformation onto
        the image plane.
        ***

        :param xyz      :   The point in which the displacement is desired
        :param dXY_l    :   Image plane displacement from the left camera
        :param dXY_r    :   Image plane displacement from the right camera
        :return         :   Optimized (dx, dy, dz) at a specific interest point
        """
        dxyz_hat = np.array(((dXY_l[0] + dXY_r[0]) / 2,
                             (dXY_l[1] + dXY_r[1]) / 2,
                             0), dtype=np.float64)

        result = least_squares(self.__inverse_displacement_transform, x0=dxyz_hat, method='trf',
                               xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy',
                               args=(xyz, dXY_l, dXY_r))

        return result.x

    def projection_object_to_image(self, point, camera_name):
        """
        Function to project points on the object plane to the corresponding point on the
        image plane using the third order polynomial.

        :param point            :      The object plane point being transformed into image plane
        :param camera_name      :      The name of the camera which is either right or left
        :return:                        The transformed point on the image plane
        """
        assert camera_name.lower() == "right" or "left", f"Unknown camera name: {camera_name}, choose either right " \
                                                         f"or left as name."

        if camera_name.lower() == "right":
            params = self.__right_calibrate_coeff
        else:
            params = self.__left_calibrate_coeff

        # Transformation Coefficient through Calibration
        coeff_x = params[:, 0]
        coeff_y = params[:, 1]
        assert len(coeff_x) and len(coeff_y) == self.__NUM_PARAM, "Coefficient must contain 19 values."

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
