import os, re, cv2
import numpy as np

import src.calibrate_least_square as nsl


def get_sorted_corner(corners):
    """
    Sort the corners for calibration --> To establish correspondence as both calibration
    and distorted image must follow the sorting algorithm

    :param corners  :   The detected corners (list)
    :return         :   A sorted list of corners (list)
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

    :param img                  :   Image of the checkerboard
    :param detected_corners     :   The detected corners using the Harris corner algorithm
    :param handle_mouse_bool    :   Whether user input is needed (True if yes else no)
    :return                     :   A new list of corners with the added/deleted corners
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img.copy()
    for corner in detected_corners:
        x, y = np.intp(corner)
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Corners', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Corners', img_copy)

    # Sub_Pixel Accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if handle_mouse_bool:
        def handle_mouse(event, x, y, flags, param):
            nonlocal img_copy, detected_corners

            if event == cv2.EVENT_LBUTTONDOWN:
                corners = np.array([[[x, y]]], dtype=np.float32)
                refined_point = cv2.cornerSubPix(gray, corners, winSize=(5, 5), zeroZone=(-1, -1),
                                                 criteria=criteria)
                refined_point = refined_point[0, 0]
                detected_corners = np.vstack([detected_corners, refined_point])

                cv2.circle(img_copy, np.intp(refined_point), 5, (0, 0, 255), -1)
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

    :param image_set_path   :   Path to input image set
    :return                 :   A list of corners of the checkerboard
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


class CalibrationPointDetector:
    def __init__(self, path_left, path_right, save_path, num_square, shape=(40, 40)):
        assert os.path.exists(path_left), f"Given path: {path_left}, does not exist."
        assert os.path.exists(path_right), f"Given path: {path_right}, does not exist."
        assert os.path.exists(save_path), f"Given path: {save_path}, does not exist."
        assert num_square > 0, "Number of squares must be positive, non-zero integers."
        assert all(dim > 0 for dim in shape), "Dimensions of squares must be positive, non-zero integers."

        self._path_left = path_left
        self._path_right = path_right
        self._save_path = save_path
        self._num_square = num_square
        self._left_image_set = self._find_all_calibration_image(self._path_left)
        self._right_image_set = self._find_all_calibration_image(self._path_right)
        self._calibrate_shape = shape

    def _object_plane_param(self, z_coords, x_offset=0, y_offset=0, dx=None, dy=None):
        """
        Create an object plane checkerboard --> Situation where image plane is parallel to object plane.
        Used as the ground truth for calibration purposes.

        :param z_coords :   The displacement in the z-direction (z_pos during calibration)
        :param x_offset :   Offset in x from the origin of image (top left)
        :param y_offset :   Offset in y from the origin of image (top left)
        :param dx       :   The displacement in the x direction
        :param dy       :   The displacement in the y direction
        :return         :   A list of object plane point (x, y, z)
        """

        num_grid = self._num_square + 1
        step_x, step_y = self._calibrate_shape

        if dx is None:
            dx = np.zeros(len(z_coords))
        if dy is None:
            dy = np.zeros(len(z_coords))

        calibrate_coordinates = []
        for i, z_coord in enumerate(z_coords):
            coordinates = [
                [x + dx[i] + x_offset, y + dy[i] + y_offset, z_coord]
                for x in np.arange(0, num_grid * step_x, step_x)
                for y in np.arange(0, num_grid * step_y, step_y)
            ]
            calibrate_coordinates.extend(coordinates)

        return np.array(calibrate_coordinates)

    def get_right_param(self):
        """
        Get the detected corner on the right image

        :return :   Right corners
        """
        right_corners = detect_corners(self._right_image_set[:, 0])
        np.save(os.path.join(self._save_path, "right_camera_pos.npy"), right_corners)
        return right_corners

    def get_left_param(self):
        """
        Get the detected corner on the left image

        :return :   Right corners
        """
        left_corners = detect_corners(self._left_image_set[:, 0])
        np.save(os.path.join(self._save_path, "left_camera_pos.npy"), left_corners)
        return left_corners

    def get_initial_calibrate_corners(self, x_offset=0, y_offset=0, dx=None, dy=None):
        """
        Get the initial 3D coordinate for evaluation of algorithm

        :param x_offset :   Offset from the origin of the image
        :param y_offset :   Offset from the origin of the image
        :param dx       :   The x displacement
        :param dy       :   The y displacement
        :return         :   Actual 3D coordinates after the displacement
        """

        original_pts = self._object_plane_param(self._left_image_set[:, 1],
                                                x_offset, y_offset, dx, dy)
        if dx == dy is None:
            np.save(os.path.join(self._save_path, "3D_camera_pos.npy"), original_pts)
        else:
            np.save(os.path.join(self._save_path, "3D_true_camera_pos.npy"), original_pts)

        return original_pts

    def _find_all_calibration_image(self, path):
        """
        Find all the calibration images within a folder and extract the depth information from the name.
        Purpose: Used to find Soloff polynomial to map 3D coordinates to 2D coordinates in image plane.

        :param path :   Path of the calibration image path
        :return     :   Two set of calibration images path (Left and right camera)
        """
        image_filenames = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_filenames.append(os.path.join(root, file))

        image_set = []
        for filename in image_filenames:
            reduced_filename = os.path.basename(filename.replace('\\', '/'))
            match = re.search(r'(-?\d+(?:\.\d+)?)mm', reduced_filename)
            if match:
                image_set.append([filename, float(match.group(1))])
        return np.array(image_set, dtype=object)

    def run_calibration(self):
        """
        Main driver for the execution of the checkerboard detection object

        :return : Save the calibration coefficient of left and right camera to the same folder
                  as the calibration images.
        """

        calibrated_point = np.array(self._object_plane_param(self._left_image_set[:, 1]))

        print("## Detecting Calibration Corners ##")
        flattened_left = detect_corners(self._left_image_set[:, 0])
        flattened_right = detect_corners(self._right_image_set[:, 0])

        assert len(calibrated_point) == len(flattened_left) == len(flattened_right), \
            "Length of calibration and distortion from either left or right camera are not equal."

        np.save(os.path.join(self._save_path, "calibrate_camera_pt.npy"), calibrated_point, allow_pickle=True)
        np.save(os.path.join(self._save_path, "left_camera_pt.npy"), flattened_left, allow_pickle=True)
        np.save(os.path.join(self._save_path, "right_camera_pt.npy"), flattened_right, allow_pickle=True)

        left_nsl_object = nsl.CalibrationTransformation(calibrated_points=calibrated_point,
                                                        distorted_points=flattened_left)
        right_nsl_object = nsl.CalibrationTransformation(calibrated_points=calibrated_point,
                                                         distorted_points=flattened_right)

        print("## Calculating transformation coefficient for left camera. ##")
        left_nsl_object.calibrate_least_square()

        print("## Calculating transformation coefficient for right camera. ##")
        right_nsl_object.calibrate_least_square()

        print(f"## Saving coefficient to the following path: {self._save_path} ##")
        left_nsl_object.save_calibration_coefficient(self._save_path, "left_cam_coeff")
        right_nsl_object.save_calibration_coefficient(self._save_path, "right_cam_coeff")

        print("## Completed the calculation of transformation coefficients ##")


if __name__ == "__main__":
    
    num_grid = 10
    left = "C:/Users/Kun Qiu/PycharmProjects/Velocity_Approx/Tool_3D/Calibration/Left"
    right = "C:/Users/Kun Qiu/PycharmProjects/Velocity_Approx/Tool_3D/Calibration/Right"
    save_path = "C:/Users/Kun Qiu/Downloads"
    num = 10
    dim_grid = (40, 40)
    
    detector = CalibrationPointDetector(
        left, right, save_path, 
        num, dim_grid
        )
    
    detector.run_calibration()
