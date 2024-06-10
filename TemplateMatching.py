import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import Utility.Polygon as pl
import Utility.Template as tm
import matplotlib.pyplot as plt
import copy


def match_template(image, template, intersection, polygon, dx=2 / 3):
    """
    Using OpenCV template matching module, the template is being matched to the
    image to find location where the similarity is the strongest
    :param  image: The image in which the template matching is applied toward
    :param  template: The template in which the source image is compared against
    :param  intersection: The spatial position of the intersection point
    :param  polygon: The spatial constraint for filtering out false positive detections
    :param  dx: Window where the non-max suppression is applied toward
    :return: List of (x, y) coordinates where similarity is above threshold value
    """
    W, H = template.shape[:2]
    rects = []

    gray_source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) \
        if len(template.shape) == 3 else template

    img_blur = cv2.GaussianBlur(gray_source, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # template_blur = cv2.GaussianBlur(gray_template, (5, 5), 0)
    _, template_thresh = cv2.threshold(gray_template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    match = cv2.matchTemplate(image=img_thresh,
                              templ=template_thresh,
                              method=cv2.TM_CCOEFF_NORMED)

    # Minimum correlation threshold
    thresh = 0.35
    temp_y, temp_x = np.where(match >= thresh)

    for (x, y) in zip(temp_x, temp_y):
        rects.append((x, y, x + dx * H, y + dx * W))

    # Apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    inter_pos = []

    for (startX, startY, endX, endY) in pick:
        x_inter = int(startX + intersection[0])
        y_inter = int(startY + intersection[1])
        if polygon.contains(pl.Point(x_inter, y_inter)):
            inter_pos.append((x_inter, y_inter))

    return inter_pos


def moving_average_validation(vectors, m, n, threshold=2.0):
    validated_vectors = np.copy(vectors)
    rows, cols = vectors.shape[:2]
    for i in range(rows):
        for j in range(cols):
            neighborhood = vectors[max(0, i - m):min(rows, i + m + 1), max(0, j - n):min(cols, j + n + 1)]
            avg_vector = np.mean(neighborhood, axis=(0, 1))
            if np.linalg.norm(vectors[i, j] - avg_vector) > threshold * np.linalg.norm(avg_vector):
                validated_vectors[i, j] = avg_vector
    return validated_vectors


def average_filter(vectors, m, n):
    smoothed_vectors = np.copy(vectors)
    rows, cols = vectors.shape[:2]
    for i in range(rows):
        for j in range(cols):
            neighborhood = vectors[max(0, i - m):min(rows, i + m + 1), max(0, j - n):min(cols, j + n + 1)]
            avg_vector = np.mean(neighborhood, axis=(0, 1))
            smoothed_vectors[i, j] = avg_vector
    return smoothed_vectors


class TemplateMatcher:
    def __init__(self, source_image_path, target_image_path,
                 template_path=None, intersection_path=None):
        """
        Class initialization for the template matcher
        :param source_image_path: Path to source image
        :param target_image_path: Path to target image
        :param template_path: Path to template image (Default is None)
        :param intersection_path: Path to intersection txt (Default is None)
        :return A Template Matching Object
        """

        self._source = cv2.imread(source_image_path)
        self._target = cv2.imread(target_image_path)

        self._template = None
        self._intersection = None

        if template_path is not None or intersection_path is not None:
            self._template = cv2.imread(template_path)
        else:
            template_object = tm.Template(source_image_path)
            template = template_object.run()
            self._template = template[0]
            self._intersection = template[1]

        """
        Array contains the following information:
        [[(x_s, y_s)], [(x_t, y_t)]] where s signifies position at source image
                                           t signifies position at target image  
        """
        self._source_points = None
        self._target_points = None
        self._polygon = None

    def set_boundary(self):
        """
        For easier detection of the relevant points, the user is asked to plot the vertices of the polygon
        that encompasses the region of interest. The boundary and the edges of the polygon is obtained to
        assist the filtering of the detected intersection.
        :return: A boundary object
        """
        points = []
        img_clone = self._source.copy()

        def pick(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append(pl.Point(x, y))
                cv2.circle(img_clone, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Boundary", img_clone)

        cv2.namedWindow("Boundary", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Boundary", 600, 600)
        cv2.imshow("Boundary", img_clone)
        cv2.setMouseCallback("Boundary", pick)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # press 'q' to quit picking points
                break
        cv2.destroyAllWindows()

        return pl.Polygon(points)

    def visualize_match(self, source_points, target_points, source_correspondence):
        """
        Visualize the displacement
        :param source_points: List of tuples [(x, y), (x1, y1), ...] representing source points
        :param target_points: List of tuples [(x, y), (x1, y1), ...] representing target points
        :param source_correspondence: List of tuples that represents the displacement
        :return: None
        """

        # Create a plot
        plt.figure(figsize=(10, 8))
        plt.imshow(self._source)

        # Plot the source points
        source_x = [point[0] for point in source_points]
        source_y = [point[1] for point in source_points]
        plt.scatter(source_x, source_y, c='red', label='Source Points', s=40, edgecolor='black')

        # Plot the target points
        target_x = [point[0] for point in target_points]
        target_y = [point[1] for point in target_points]
        plt.scatter(target_x, target_y, c='blue', label='Target Points', s=40, edgecolor='black')

        # Plot vectors with validation
        for correspondence in source_correspondence:
            start_point = correspondence[0]
            plt.arrow(start_point[0], start_point[1], correspondence[1], correspondence[2],
                      head_width=2, head_length=4, fc='red', ec='red')

        plt.legend()
        plt.title("Approximation of the intersection position")
        plt.axis('off')
        plt.show()

    def correspondence_position(self, source, target):
        """
        Find the correspondence between two set of intersection points
        on the image
        :param source: Intersections within the source image
        :param target: Intersections within the target image
        :return: A correspondence between the intersection points
        """

        distance_thresh = 0.5 * self.get_length()
        target_tmp = copy.deepcopy(target)

        correspondence = []
        for source_point in source:
            dist_source = []
            for target_point in target_tmp:
                dist_source.append(np.sqrt((target_point[0] - source_point[0]) ** 2 +
                                           (target_point[1] - source_point[1]) ** 2))

            # Find the minimum distance for the current source point
            min_index = np.argmin(dist_source)
            if dist_source[min_index] <= distance_thresh:
                x_displace = target_tmp[min_index][0] - source_point[0]
                y_displace = target_tmp[min_index][1] - source_point[1]
                correspondence.append((source_point, x_displace, y_displace))
                target_tmp.pop(min_index)

        return correspondence

    def match_template_driver(self):
        """
        Driver function for the template matching algorithm using the OpenCV Module
        :return: None
        """
        self._polygon = self.set_boundary()
        self._source_points = match_template(self._source, self._template,
                                             self._intersection, self._polygon)
        self._target_points = match_template(self._target, self._template,
                                             self._intersection, self._polygon)

        if not self._source_points or not self._target_points:
            print("No points detected in source or target.")

        displacement_field = self.correspondence_position(self._source_points, self._target_points)
        self.visualize_match(self._source_points, self._target_points, displacement_field)

    def get_length(self):
        return self._intersection[2]

    def get_boundary(self):
        return self._polygon


def main():
    parser = argparse.ArgumentParser(description="Template Matching")
    parser.add_argument("source_image", help="Path to the source image")
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("template", nargs='?', default=None, help="Path to the template image (optional)")
    parser.add_argument("intersection", nargs='?', default=None, help="Path to the intersection file (optional)")
    args = parser.parse_args()

    """
    Sample visualization of the result of template matching on the image
    """
    matcher = TemplateMatcher(args.source_image, args.target_image,
                              args.template, args.intersection)
    matcher.match_template_driver()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
