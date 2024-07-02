import numpy as np
import matplotlib.pyplot as plt
import cv2
import non_linear_least_square as nnl


class CornerDetector:
    def __init__(self, distorted_img):
        self.img = cv2.imread(distorted_img)
        # self.calibrate_img = cv2.imread(calibration_img)
        self.corners = None
        self._optimized_param = None

    def detect_corners(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        self.corners = corners

    def visualize_corners(self, handle_mouse_bool=False):
        img_copy = self.img.copy()
        # Draw corners on the image
        for corner in self.corners:
            x, y = np.intp(corner)
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)  # Green dots for refined corners

        cv2.namedWindow('Corners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Corners', 512, 512)
        cv2.imshow('Corners', img_copy)

        if handle_mouse_bool:
            def handle_mouse(event, x, y, flags, param):
                nonlocal img_copy

                if event == cv2.EVENT_LBUTTONDOWN:
                    self.corners = np.vstack([self.corners, [x, y]])
                    cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow('Corners', img_copy)

                elif event == cv2.EVENT_RBUTTONDOWN:
                    # Check distance to each existing corner to determine if clicked
                    for i, corner in enumerate(self.corners):
                        dist = np.sqrt((x - corner[0]) ** 2 + (y - corner[1]) ** 2)
                        if dist <= 5:  # 5 pixels threshold for clicking on a corner
                            self.corners = np.delete(self.corners, i, axis=0)
                            img_copy = self.img.copy()  # Reset image copy
                            for corner in self.corners:
                                cv2.circle(img_copy, np.intp(corner), 5, (0, 255, 0), -1)
                            cv2.imshow('Corners', img_copy)

            # Create a resizable named window
            cv2.setMouseCallback('Corners', handle_mouse)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_sorted_corner(self):
        sorted_x_corners = sorted(self.corners, key=lambda p: p[0])
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


# Example usage:
if __name__ == "__main__":
    detector = CornerDetector('../3D_Experiment/Calibration/right_35deg.png')
    detector.detect_corners()
    detector.visualize_corners(handle_mouse_bool=True)
    distorted_corners = detector.get_sorted_corner()

    source_detector = CornerDetector('../3D_Experiment/Calibration/right_0deg.png')
    source_detector.detect_corners()
    source_detector.visualize_corners(handle_mouse_bool=True)
    calibrate_corners = source_detector.get_sorted_corner()

    distorted_corners = np.array(distorted_corners, dtype=np.intp)
    calibrated_corners = np.array(calibrate_corners, dtype=np.intp)

    # Plotting the correspondences
    plt.figure(figsize=(10, 5))
    plt.title('Correspondences between Distorted and Calibrate Corners')

    # Plot distorted corners
    plt.scatter(distorted_corners[:, 0], distorted_corners[:, 1], c='r', label='Distorted Corners')
    # Plot calibrate corners
    plt.scatter(calibrate_corners[:, 0], calibrate_corners[:, 1], c='b', label='Calibrate Corners')

    # Draw lines between corresponding points
    for i in range(len(distorted_corners)):
        plt.plot([distorted_corners[i, 0], calibrate_corners[i, 0]],
                 [distorted_corners[i, 1], calibrate_corners[i, 1]], 'g-')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    param_object = nnl.NonLinearLeastSquare(calibrated_corners, distorted_corners)
    param_op = param_object.calculate_least_square()
