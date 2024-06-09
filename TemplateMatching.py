import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import Utility.Polygon as pl
import Utility.Template as tm
import matplotlib.pyplot as plt


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

    template_blur = cv2.GaussianBlur(gray_template, (5, 5), 0)
    _, template_thresh = cv2.threshold(template_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    match = cv2.matchTemplate(image=img_thresh,
                              templ=template_thresh,
                              method=cv2.TM_CCOEFF_NORMED)

    # Minimum correlation threshold
    thresh = 0.4
    temp_y, temp_x = np.where(match >= thresh)

    for (x, y) in zip(temp_x, temp_y):
        rects.append((x, y, x + dx * H, y + dx * W))

    # Apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    print("[INFO] {} matched locations *after* NMS".format(len(pick)))

    inter_pos = []
    for (startX, startY, endX, endY) in pick:
        x_inter = int(startX + intersection[0])
        y_inter = int(startY + intersection[1])
        if polygon.contains(pl.Point(x_inter, y_inter)):
            inter_pos.append((x_inter, y_inter))

    return inter_pos


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
            template_object = tm.Template(source_image_path, 2)
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

        # self._matchedXCoordPreNMS = []
        # self._matchedYCoordPreNMS = []
        # self._matchedXCoordPostNMS = []
        # self._matchedYCoordPostNMS = []

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
                cv2.imshow("image", img_clone)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        cv2.imshow("image", img_clone)
        cv2.setMouseCallback("image", pick)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # press 'q' to quit picking points
                break
        cv2.destroyAllWindows()

        return pl.Polygon(points)

    def visualize_match(self, source_points, target_points):
        """
        Visualize the match after NMS
        :param source_points: List of tuples [(x, y), (x1, y1), ...] representing source points
        :param target_points: List of tuples [(x, y), (x1, y1), ...] representing target points
        :return: None
        """

        # Convert the image from BGR to RGB (since OpenCV loads images in BGR)
        image_rgb = cv2.cvtColor(self._source, cv2.COLOR_BGR2RGB)

        # Create a plot
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)

        # Plot the source points
        if source_points:
            source_x, source_y = zip(*source_points)
            plt.scatter(source_x, source_y, c='red', label='Source Points', s=40, edgecolor='black')

        # Plot the target points
        if target_points:
            target_x, target_y = zip(*target_points)
            plt.scatter(target_x, target_y, c='blue', label='Target Points', s=40, edgecolor='black')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.title("Approximation of the intersection position")
        plt.axis('off')  # Hide the axes
        plt.show()

    # def matching_displacement(self, source, target):
    #     """
    #     Find the correspondence between two set of intersection points
    #     on the image
    #     :param source: Intersections within the source image
    #     :param target: Intersections within the target image
    #     :return: A correspondence between the intersection points
    #     """
    #     distance_thresh = 0.25 * self.get_length()
    #
    #     # Construct arrays of coordinates for each point
    #     source = np.column_stack((self._matchedXCoordPostNMS,
    #                               self._matchedYCoordPostNMS))
    #     target = np.column_stack((target.get_x_coord(),
    #                               target.get_y_coord()))
    #
    #     correspondence = []
    #     for x in source:
    #         dist_x = []
    #         for y in target:
    #             dist_x.append(np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2))
    #
    #         min_index = np.argmin(dist_x)  # Find the minimum distance for the current source point
    #         if dist_x[min_index] < distance_thresh:
    #             correspondence.append([x, target[min_index]])
    #
    #     return correspondence

    def match_template_driver(self):
        """
        Driver function for the template matching algorithm using the OpenCV Module
        :return: None
        """
        self._polygon = self.set_boundary()
        self._source_points = match_template(self._source, self._template,
                                             self._intersection, self._polygon)
        self._target_points = match_template(self._target, self._template,
                                             self._intersection, self._template)

        self.visualize_match(self._source_points, self._target_points)

    # def visualizeMatchBeforeNMS(self):
    #     """
    #     Visualize the match before NMS
    #     :return: Visualize the match
    #     """
    #     clone = self._source.copy()
    #     print("[INFO] {} matched locations *before* NMS".format(len(self._matchedYCoordPreNMS)))
    #
    #     for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
    #         cv2.circle(clone, (int(x + self._intersection[0]),
    #                            int(y + self._intersection[1])),
    #                    2, (0, 255, 0), -1)
    #
    #     cv2.namedWindow("Before NMS", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("Before NMS", 800, 600)
    #     cv2.imshow("Before NMS", clone)
    #     cv2.waitKey(0)

    # def NonMaxSuppression(self, dx=2 / 3):
    #     """
    #     Reduce the amount of overlapping points in the template matching process
    #     :param dx: size of the rectangle to check for overlap
    #     :return: None
    #     """
    #
    #     W, H = self._template.shape[:2]
    #     rects = []
    #
    #     for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
    #         rects.append((x, y, x + dx * H, y + dx * W))
    #
    #     # apply non-maxima suppression to the rectangles
    #     pick = non_max_suppression(np.array(rects))
    #     print("[INFO] {} matched locations *after* NMS".format(len(pick)))
    #
    #     for (startX, startY, endX, endY) in pick:
    #         x_inter = int(startX + self._intersection[0])
    #         y_inter = int(startY + self._intersection[1])
    #         if self._polygon.contains(pl.Point(x_inter, y_inter)):
    #             self._matchedXCoordPostNMS.append(x_inter)
    #             self._matchedYCoordPostNMS.append(y_inter)

    # def visualizeMatchAfterNonMaxSuppression(self, source_points, target_points):
    #     """
    #     Visualize the match after NMS
    #     :param source_points
    #     :param target_points
    #     :return: None
    #     """
    #
    #     clone = self._source.copy()
    #     for i in range(len(self._matchedXCoordPostNMS)):
    #         cv2.circle(clone, (self._matchedXCoordPostNMS[i], self._matchedYCoordPostNMS[i]),
    #                    2, (0, 255, 0), -1)
    #     cv2.namedWindow("After NMS", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("After NMS", 800, 600)
    #     cv2.imshow("After NMS", clone)
    #     cv2.waitKey(0)

    # def get_x_coord(self):
    #     return self._matchedXCoordPostNMS
    #
    # def get_y_coord(self):
    #     return self._matchedYCoordPostNMS

    def get_length(self):
        return self._intersection[2]

    def get_boundary(self):
        return self._polygon

    # def run(self):
    #     """
    #     Run function for the template matching
    #     :return: None
    #     """
    #     self.set_boundary()
    #     self.match_template()
    #     self.visualizeMatchBeforeNMS()
    #     self.visualizeMatchAfterNonMaxSuppression()
    #     cv2.destroyAllWindows()


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
    matcher.set_boundary()
    matcher.match_template_driver()
    # matcher.visualizeMatchBeforeNMS()
    # matcher.visualizeMatchAfterNonMaxSuppression()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
