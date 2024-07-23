import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from calibration_transform_coefficient import CalibrationPointDetector
from calibration_transform_coefficient import detect_corners


def displacement_function(XY_0, XY_t_l, XY_t_r, chunk_size):
    """
    Calculate the displacement in the image plane

    :param XY_0         : Initial positions of the points in the image plane
    :param XY_t_l       : Position of the final calibration position in image plane of left cam
    :param XY_t_r       : Position of the final calibration position in image plane of right cam
    :param chunk_size   : Size of the set of calibration data (original is flattened)
    :return             : The displacement in the image plane (dX, dY)
    """

    assert XY_t_l.shape == XY_t_r.shape, "Shape of detected points from left and right camera are not equal: " \
                                         f"left camera: {XY_t_l.shape}, right camera: {XY_t_r.shape}"
    num_chunk = len(XY_t_l) // chunk_size

    XY_0 = np.array([XY_0[i:i + chunk_size] for i in range(0, len(XY_0), chunk_size)])
    XY_t_l = np.array([XY_t_l[i:i + chunk_size] for i in range(0, len(XY_t_l), chunk_size)])
    XY_t_r = np.array([XY_t_r[i:i + chunk_size] for i in range(0, len(XY_t_r), chunk_size)])

    for i in range(num_chunk):
        XY_t_l[i, :, :] = XY_t_l[i, :, :] - XY_0[0]
        XY_t_r[i, :, :] = XY_t_r[i, :, :] - XY_0[1]

    return np.array(XY_t_l), np.array(XY_t_r)


class CalibrationDisplacementMatrix(CalibrationPointDetector):
    def __init__(self, path, num_square, shape=(40, 40)):
        super().__init__(path, num_square, shape)
        self._num_points = pow((num_square + 1), 2)
        self._origin_path = np.array((None, None))
        self._transformation_matrix = np.zeros((self._num_points, 4, 3))
        self._object_pos = self._object_plane_param([0])

    def __object_plane_displace(self, dx, dy, dz):
        """
        Create an object plane checkerboard --> Situation where image plane is parallel to object plane.
        Used as the ground truth for calibration purposes.

        :param dx   :   The displacement in the x direction
        :param dy   :   The displacement in the y direction
        :param dz   :   The displacement in the z direction
        :return     :   A list of object plane point (dx, dy, dz)
        """

        num_grid = self._num_square + 1
        calibrate_displacement = np.array((dx, dy, dz))
        return np.tile(calibrate_displacement, (pow(num_grid, 2), 1))

    def __find_all_calibration_image(self):
        """
        Find all the calibration images within a folder and extract the displacement information from the name.
        Purpose: Used to find Soloff polynomial to map 3D displacement to 2D displacement in image plane.

        :return     :   Two set of calibration images path (Left and right camera)
        """
        image_filenames = []
        for root, _, files in os.walk(self._path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_filenames.append(os.path.join(root, file))

        left_image_set, right_image_set = [], []
        for filename in image_filenames:
            reduced_filename = os.path.basename(filename.replace('\\', '/'))

            if "(0,0,0)" in filename.lower():
                if 'left' in filename.lower():
                    self._origin_path[0] = filename
                elif 'right' in filename.lower():
                    self._origin_path[1] = filename
            else:
                match = re.search(r'\(([^)]+)\)', reduced_filename)
                if match:
                    try:
                        numbers = [int(x) for x in match.group(1).split(',')]
                        if len(numbers) == 3:
                            if 'left' in filename.lower():
                                left_image_set.append([filename, numbers])
                            elif 'right' in filename.lower():
                                right_image_set.append([filename, numbers])
                    except ValueError:
                        print(f"Invalid number format in filename: {filename}")

        return np.array(left_image_set, dtype=object), np.array(right_image_set, dtype=object)

    def run_calibration(self):
        """
        Main driver for the execution of the checkerboard detection object

        :return : Save the calibration coefficient of left and right camera to the same folder
                  as the calibration images.
        """

        self._left_image_set, self._right_image_set = self.__find_all_calibration_image()

        print("## Detecting Calibration Corners ##")
        flattened_origin = detect_corners(self._origin_path).astype(float)
        flattened_left = detect_corners(self._left_image_set[:, 0]).astype(float)
        flattened_right = detect_corners(self._right_image_set[:, 0]).astype(float)

        # flattened_origin = np.load(os.path.join(self._path, "source_pt.npy"), allow_pickle=True).astype(float)
        # flattened_left = np.load(os.path.join(self._path, "left_camera_pt.npy"), allow_pickle=True).astype(float)
        # flattened_right = np.load(os.path.join(self._path, "right_camera_pt.npy"), allow_pickle=True).astype(float)

        assert len(flattened_left) == len(flattened_right), \
            "Length of distortion images from either left or right camera are not equal."

        np.save(os.path.join(self._path, "object_pt.npy"), self._object_pos, allow_pickle=True)
        # np.save(os.path.join(self._path, "source_pt.npy"), flattened_origin, allow_pickle=True)
        # np.save(os.path.join(self._path, "left_camera_pt.npy"), flattened_left, allow_pickle=True)
        # np.save(os.path.join(self._path, "right_camera_pt.npy"), flattened_right, allow_pickle=True)

        dis_l, dis_r = displacement_function(flattened_origin, flattened_left, flattened_right, self._num_points)

        def transformation_error(params, dxyz, dXY_l, dXY_r):
            F = params.reshape(4, 3)  # Transformation matrix
            transformed_displacement = F @ dxyz.T  # Apply transformation
            error_l = transformed_displacement[0:2] - dXY_l
            error_r = transformed_displacement[2:4] - dXY_r
            return np.concatenate((error_l.flatten(), error_r.flatten()))

        num_calibrations, num_points = dis_l.shape[0], dis_l.shape[1]
        transformation_matrix = np.zeros((num_points, 4, 3))

        dxyz = np.array(self._left_image_set[:, 1])

        for idx_cal in range(num_calibrations):
            dXY_l = dis_l[idx_cal]
            dXY_r = dis_r[idx_cal]

            initial_guess = np.zeros((num_points, 12))
            results = np.array([least_squares(transformation_error, x0=initial_guess[i], method='trf',
                                              xtol=1.e-15, gtol=1.e-15, ftol=1.e-15, loss='cauchy',
                                              args=(np.array(dxyz[idx_cal]), dXY_l[i], dXY_r[i])).x.reshape(4, 3)
                                for i in range(num_points)])

            transformation_matrix += results

        np.save(os.path.join(self._path, "transformation_matrix.npy"), transformation_matrix, allow_pickle=True)

    def visualize_coefficients(self, coefficient_idx, show=False):
        """
        Visualize the transformation matrix for each coefficient on the x, y, z coordinate

        :param coefficient_idx  :   The index of the coefficient plotted
        :param show             :   Whether to display plot
        :return                 :   Visualize the plot of magnification coefficient
        """
        row, col = coefficient_idx
        assert row < 4, f"Row {row} does not exist in the transformation matrix."
        assert col < 3, f"Column {col} does not exist in the transformation matrix."

        coeff = self._transformation_matrix[:, row, col]

        plt.figure(figsize=(10, 5))
        plt.scatter(self._object_pos[:, 0], self._object_pos[:, 1], c=coeff, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Coefficient 1')
        plt.title('First Coefficient of Transformation Matrix')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        if show:
            plt.show()


# Example usage:
if __name__ == "__main__":
    """
    Example execution of the class.
    """
    num_grid = 10
    directory_dis = '../3D_Experiment/Calibration/Displacement'
    detector_dis = CalibrationDisplacementMatrix(directory_dis, num_grid, (40, 40))
    detector_dis.run_calibration()
    # detector_dis.visualize_coefficients((3, 1), True)
