import numpy as np
import re
import cv2
import calibrate_least_square as nsl
import os


def get_sorted_corner(corners):
    """
    Sort the corners for calibration --> To establish correspondence as both calibration
    and distorted image must follow the sorting algorithm

    :param corners:     The detected corners (list)
    :return:            A sorted list of corners (list)
    """
    sorted_x_corners = sorted(corners, key=lambda p: p[0])
    sorted_corners = []

    index = 0
    cur_x_val = sorted_x_corners[0][0]
    for i in range(len(sorted_x_corners)):
        x, _ = sorted_x_corners[i]
        if np.abs(x - cur_x_val) >= 5:
            if x != cur_x_val:
                sorted_corners.extend(sorted(sorted_x_corners[index:i], key=lambda p: p[1]))
                index = i
                cur_x_val = x

    sorted_corners.extend(sorted(sorted_x_corners[index:], key=lambda p: p[1]))

    return sorted_corners


def manual_detection_corners(img, detected_corners, handle_mouse_bool=False):
    """
    The detection algorithm using Harris corner might not capture all the corners of the
    checkerboard. Allow user to input and delete points.

    :param img:                 Image of the checkerboard
    :param detected_corners:    The detected corners using the Harris corner algorithm
    :param handle_mouse_bool:   Whether user input is needed (True if yes else no)
    :return:                    A new list of corners with the added/deleted corners
    """
    img_copy = img.copy()
    for corner in detected_corners:
        x, y = np.intp(corner)
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Corners', 512, 512)
    cv2.imshow('Corners', img_copy)

    if handle_mouse_bool:
        def handle_mouse(event, x, y, flags, param):
            nonlocal img_copy, detected_corners

            if event == cv2.EVENT_LBUTTONDOWN:
                detected_corners = np.vstack([detected_corners, [x, y]])
                cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Corners', img_copy)

            elif event == cv2.EVENT_RBUTTONDOWN:
                for i, corner in enumerate(detected_corners):
                    dist = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
                    if dist <= 5:
                        detected_corners = np.delete(detected_corners, i, axis=0)
                        img_copy = img.copy()
                        for corner in detected_corners:
                            cv2.circle(img_copy, np.intp(corner), 5, (0, 255, 0), -1)
                        cv2.imshow('Corners', img_copy)

        cv2.setMouseCallback('Corners', handle_mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_corners


def detect_corners(image_set_path):
    """
    Detect the corner to subpixel accuracy with the Harris Corner Detection algorithm provided
    by OpenCV

    :param image_set_path:      Path to the image
    :return:                    A list of corners of the checkerboard
    """
    assert isinstance(image_set_path, np.ndarray), "Incorrect array structure: requires Numpy array."

    all_img_corners = []
    for image in image_set_path:
        img = cv2.imread(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (11, 11), (-1, -1), criteria)
        single_img_corners = manual_detection_corners(img, corners, handle_mouse_bool=True)
        sorted_corners = np.array(get_sorted_corner(single_img_corners))

        i = 0
        while i < len(sorted_corners) - 1:
            if np.sqrt((sorted_corners[i, 0] - sorted_corners[i + 1, 0]) ** 2 +
                       (sorted_corners[i, 1] - sorted_corners[i + 1, 1]) ** 2) < 1:
                sorted_corners = np.delete(sorted_corners, i + 1, axis=0)
            else:
                i += 1
        all_img_corners.extend(sorted_corners)

    all_img_corners = np.array(all_img_corners, dtype=object)
    return all_img_corners


class CornerDetector:
    def __init__(self, image_set_path, num_square, calibrate_shape=(10, 10)):
        self.__path = image_set_path
        self.__num_square = num_square

        image_filenames = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_filenames.append(os.path.join(root, file))

        self.__left_image_set = []
        self.__right_image_set = []
        for filename in image_filenames:
            match = re.search(r'([-]?\d+)mm', filename)
            if match:
                if 'left' in filename.lower():
                    self.__left_image_set.append([filename, int(match.group(1))])
                elif 'right' in filename.lower():
                    self.__right_image_set.append([filename, int(match.group(1))])

        self.__left_image_set = np.array(self.__left_image_set, dtype=object)
        self.__right_image_set = np.array(self.__right_image_set, dtype=object)
        self.__calibrate_shape = calibrate_shape

    def __object_plane_coordinates(self, num_grid, z_coords):
        """
        Create an object plane checkerboard --> Situation where image plane is parallel to object plane.
        Used as the ground truth for calibration purposes.

        :param num_grid:    Number of checkerboard squares
        :param z_coords:    The z-coordinate that the plane is at during calibration
        :return:            A list of object plane point (x, y, z)
        """

        num_grid += 1
        step_x, step_y = self.__calibrate_shape

        calibrate_coordinates = []
        for z_coord in z_coords:
            coordinates = [
                [x, y, z_coord]
                for x in np.arange(0, num_grid * step_x, step_x)
                for y in np.arange(0, num_grid * step_y, step_y)
            ]
            calibrate_coordinates.extend(coordinates)

        return calibrate_coordinates

    def run(self):
        """
        Main driver for the execution of the checkerboard detection object

        :return: Save the calibration coefficient of left and right camera to the same folder
                 as the calibration images.
        """
        calibrated_point = self.__object_plane_coordinates(self.__num_square, self.__left_image_set[:, 1])
        flattened_right = detect_corners(self.__right_image_set[:, 0])
        flattened_left = detect_corners(self.__left_image_set[:, 0])

        np.save(os.path.join(self.__path, "calibrate_camera_pt.npy"), calibrated_point, allow_pickle=True)
        np.save(os.path.join(self.__path, "left_camera_pt.npy"), flattened_left, allow_pickle=True)
        np.save(os.path.join(self.__path, "right_camera_pt.npy"), flattened_right, allow_pickle=True)

        print("## Calculating transformation coefficient for left camera. ##")
        left_nsl_object = nsl.CalibrationTransformation(calibrated_points=calibrated_point,
                                                        distorted_points=flattened_left)
        left_nsl_object.calibrate_least_square()

        print("## Calculating transformation coefficient for right camera. ##")
        right_nsl_object = nsl.CalibrationTransformation(calibrated_points=calibrated_point,
                                                         distorted_points=flattened_right)
        right_nsl_object.calibrate_least_square()

        print(f"## Saving coefficient to the following path: {self.__path} ##")
        left_nsl_object.save_calibration_coefficient(self.__path, "left_cam_coeff")
        right_nsl_object.save_calibration_coefficient(self.__path, "right_cam_coeff")


# Example usage:
if __name__ == "__main__":
    """
    Example execution of the class.
    """
    directory = '../3D_Experiment/Calibration/'
    num_grid = 10
    detector = CornerDetector(directory, num_grid, (40, 40))
    detector.run()
