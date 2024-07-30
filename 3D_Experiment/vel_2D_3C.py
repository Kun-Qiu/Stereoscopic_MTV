import os

import numpy as np

import inverse_least_square as ils
from interpolator import DisplacementInterpolator


def read_numpy_file(input_var):
    """
    Read the numpy file if a path is given, else if the input is a ndarray, return
    the ndarray

    :param input_var    :   Unknown data (String or ndarray)
    :return             :   Return the content of the input_var
    """
    if isinstance(input_var, str) and os.path.exists(input_var):
        return np.load(input_var, allow_pickle=True).astype(float)
    elif isinstance(input_var, np.ndarray):
        return input_var


class Velocity2D_3C:
    def __init__(self, XY_l, XY_r, dXY_l, dXY_r, path_coeff_l, path_coeff_r, grid_density=500, window_size=44):
        self._XY_l = read_numpy_file(XY_l)
        self._XY_r = read_numpy_file(XY_r)
        self._dXY_l = read_numpy_file(dXY_l)
        self._dXY_r = read_numpy_file(dXY_r)

        self._coeff_l = np.load(path_coeff_l, allow_pickle=True)
        self._coeff_r = np.load(path_coeff_r, allow_pickle=True)
        self._inverse_obj = ils.InverseTransform(self._coeff_l, self._coeff_r)

        self._left_intp = DisplacementInterpolator(XY_l, dXY_l, grid_density=grid_density)
        self._right_intp = DisplacementInterpolator(XY_r, dXY_r, grid_density=grid_density)

        self._grid_density = grid_density
        self._window_size = window_size

    def calculate_3D_displacement(self, xyz):
        """
        Calculate the 3D displacement using Soloff Calibration method

        :param xyz  :   3D coordinate where displacement is desire
        :return     :   Array of 3D displacement at xyz
        """
        if not isinstance(xyz, np.ndarray) or xyz.ndim != 3:
            raise ValueError("xyz must be a 3D numpy array")

        displace_3D_arr = []

        XY_left, dXY_left = self._left_intp.get_interpolate()
        XY_right, dXY_right = self._right_intp.get_interpolate()

        left_dXY = self.__displacement_2D_extractor(xyz, XY_left[:, :, 0], XY_left[:, :, 1], dXY_left[:, :, 0],
                                                    dXY_left[:, :, 1], 'left').reshape(-1, 2)
        right_dXY = self.__displacement_2D_extractor(xyz, XY_right[:, :, 0], XY_right[:, :, 1], dXY_right[:, :, 0],
                                                     dXY_right[:, :, 1], 'right').reshape(-1, 2)

        for i, point in enumerate(xyz.reshape(-1, 3)):
            left_np_pt = np.array((left_dXY[i, 0], left_dXY[i, 1]))
            right_np_pt = np.array((right_dXY[i, 0], right_dXY[i, 1]))
            result = self._inverse_obj.inverse_displacement(point, left_np_pt, right_np_pt)
            displace_3D_arr.append(result)

        return np.array(displace_3D_arr).reshape(xyz.shape)

    def __displacement_2D_extractor(self, xyz, x_arr, y_arr, dx_arr, dy_arr, name):
        """

        :param xyz      :   Array of interest points
        :param x_arr    :   Interpolated X_Coordinate array
        :param y_arr    :   Interpolated Y_Coordinate array
        :param dx_arr   :   Interpolated dX array
        :param dy_arr   :   Interpolated dY array
        :param name     :   Name of the camera (left or right)
        :return         :   Array of (x, y, z, dx, dy, dz)
        """

        width, height, _ = xyz.shape
        dXY_2D = np.zeros((width, height, 2))
        n = self._window_size

        for i, column in enumerate(xyz):
            for j, point in enumerate(column):
                camera_pt = self._inverse_obj.projection_object_to_image(point, name)
                if name.lower() == "left":
                    camera_dXY = self._left_intp.compute_interpolate_point(camera_pt)
                else:
                    camera_dXY = self._right_intp.compute_interpolate_point(camera_pt)
                distances = np.sqrt((x_arr - camera_pt[0]) ** 2 + (y_arr - camera_pt[1]) ** 2)
                min_index = np.unravel_index(np.argmin(distances), distances.shape)

                min_i, min_j = min_index
                start_i = max(0, min_i - (n // 2))
                end_i = min(distances.shape[0], min_i + (n // 2) + 1)
                start_j = max(0, min_j - (n // 2))
                end_j = min(distances.shape[1], min_j + (n // 2) + 1)

                # Calculate average dx and dy within the window
                dx_window = dx_arr[start_i:end_i, start_j:end_j]
                dy_window = dy_arr[start_i:end_i, start_j:end_j]
                valid_indices = ~np.isnan(dx_window) & ~np.isnan(dy_window)
                if np.any(valid_indices):
                    avg_dx = np.mean(np.concatenate([dx_window[valid_indices], camera_dXY[0]], axis=0))
                    avg_dy = np.mean(np.concatenate([dy_window[valid_indices], camera_dXY[1]], axis=0))
                    dXY_2D[i, j] = [avg_dx, avg_dy]

        return dXY_2D

    def interpolate_3D_displacement(self, XY, dXYZ, display=False):
        """
        Interpolate the 3D displacement at several known point to a desired grid

        :param XY       :   Known coordinates
        :param dXYZ     :   Known displacement (3D)
        :param display  :   Boolean on whether to display interpolation
        :return         :   Interpolated grid and displacement
        """
        assert dXYZ.shape[2] == 3, f"Input displacement matrix have dimension of {dXYZ.shape[2]}, but a dim of 3 " \
                                   f"is required."
        if XY.shape[2] != 2:
            XY = XY[:, :, 0:2]

        intp_obj = DisplacementInterpolator(XY.reshape(-1, 2), dXYZ.reshape(-1, 3), grid_density=self._grid_density)

        if display:
            intp_obj.plot_interpolation("Displacement [mm]")

        return intp_obj.get_interpolate()
