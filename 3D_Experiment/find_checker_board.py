import numpy as np
import re
import cv2
import non_linear_least_square as nsl
import os
import matplotlib.pyplot as plt


def get_sorted_corner(corners):
    sorted_x_corners = sorted(corners, key=lambda p: p[0])
    sorted_corners = []

    index = 0
    cur_x_val = sorted_x_corners[0][0]
    for i in range(len(sorted_x_corners)):
        x, _ = np.intp(sorted_x_corners[i])
        if np.abs(x - cur_x_val) >= 5:
            if x != cur_x_val:
                sorted_corners.extend(sorted(sorted_x_corners[index:i], key=lambda p: p[1]))
                index = i
                cur_x_val = x

    # Adding the last segment of corners
    sorted_corners.extend(sorted(sorted_x_corners[index:], key=lambda p: p[1]))

    return sorted_corners


def manual_detection_corners(img, detected_corners, handle_mouse_bool=False):
    img_copy = img.copy()
    for corner in detected_corners:
        x, y = np.intp(corner)
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)  # Green dots for refined corners

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
                # Check distance to each existing corner to determine if clicked
                for i, corner in enumerate(detected_corners):
                    dist = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
                    if dist <= 5:  # 5 pixels threshold for clicking on a corner
                        detected_corners = np.delete(detected_corners, i, axis=0)
                        img_copy = img.copy()  # Reset image copy
                        for corner in detected_corners:
                            cv2.circle(img_copy, np.intp(corner), 5, (0, 255, 0), -1)
                        cv2.imshow('Corners', img_copy)

        # Create a resizable named window
        cv2.setMouseCallback('Corners', handle_mouse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detected_corners


class CornerDetector:
    def __init__(self, image_set_path, calibrate_shape=(10, 10)):
        assert len(image_set_path) >= 1, "No path is given for the calibration images."

        self.__left_image_set = []
        self.__right_image_set = []
        for filename in image_set_path:
            match = re.search(r'([-]?\d+)mm', filename)
            if match:
                if 'left' in filename.lower():
                    self.__left_image_set.append([filename, int(match.group(1))])
                elif 'right' in filename.lower():
                    self.__right_image_set.append([filename, int(match.group(1))])

        self.__left_image_set = np.array(self.__left_image_set, dtype=object)
        self.__right_image_set = np.array(self.__right_image_set, dtype=object)
        self.__calibrate_shape = calibrate_shape
        self.__optimized_param = None

    def __detect_corners(self, image_set_path):
        assert isinstance(image_set_path, np.ndarray), "Incorrect array structure: requires Numpy array."

        all_img_corners = []
        for image in image_set_path[:, 0]:
            img = cv2.imread(image)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
            dst = np.uint8(dst)

            # find centroids
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

            # define the criteria to stop and refine the corners
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
        return all_img_corners, image_set_path[:, 1]

    def __object_plane_coordinates(self, num_grid, z_coords):
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
        calibrated_point = self.__object_plane_coordinates(11, self.__left_image_set[:, 1])
        flattened_right, _ = self.__detect_corners(self.__right_image_set)
        right_nsl_object = nsl.NonLinearLeastSquare(calibrated_points=calibrated_point,
                                                    distorted_points=flattened_right)

        right_param = right_nsl_object.calculate_least_square()

        # fig, ax = plt.subplots()
        #
        # # Plot flattened_right
        # ax.scatter(flattened_right[:121, 0], flattened_right[:121, 1], color='blue', label='Flattened Right')
        #
        # # Plot predicted points
        # for pt in calibrated_point[:121]:
        #     xpredicted, ypredicted = nsl.project_object_image(19, right_param, pt)
        #     ax.scatter(xpredicted, ypredicted, color='red', label='Predicted')
        #
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        # ax.legend()
        # plt.title('Comparison of Predicted and Flattened Right Points')
        # plt.show()

        flattened_left, _ = self.__detect_corners(self.__left_image_set)


# Example usage:
if __name__ == "__main__":
    directory = '../3D_Experiment/Calibration/'
    image_filenames = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_filenames.append(os.path.join(root, file))

    detector = CornerDetector(image_filenames, (40, 40))
    detector.run()
