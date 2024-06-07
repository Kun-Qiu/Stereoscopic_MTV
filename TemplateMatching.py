import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import Utility.Polygon as pl


class TemplateMatcher:
    def __init__(self, source_image_path, template_path, intersection_path):
        """
        Class initialization for the template matcher
        :param source_image_path: path to source image
        :param template_path: path to template image
        :param intersection_path: path to intersection txt
        """

        self._source = cv2.imread(source_image_path)
        self._template = cv2.imread(template_path)
        self._intersection = np.loadtxt(intersection_path)
        self._matchedXCoordPreNMS = []
        self._matchedYCoordPreNMS = []
        self._matchedXCoordPostNMS = []
        self._matchedYCoordPostNMS = []
        self._polygon = None

    def set_boundary(self):
        """
        Obtain the boundary and the edges of the polygon
        :return: None
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
                self._polygon = pl.Polygon(points)
                break
        cv2.destroyAllWindows()

    def match_template(self):
        """
        Main driver for template matching using openCV
        Since float is being converted into int for the intersections, the error accumulates
        exponentially as the scale_num increase
        :return: None
        """
        gray_source = cv2.cvtColor(self._source, cv2.COLOR_BGR2GRAY) \
            if len(self._source.shape) == 3 else self._source

        img_blur = cv2.GaussianBlur(gray_source, (5, 5), 0)
        _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_template = cv2.cvtColor(self._template, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(image=img_thresh,
                                  templ=gray_template,
                                  method=cv2.TM_CCOEFF_NORMED)

        # Minimum correlation threshold
        thresh = 0.4
        temp_y, temp_x = np.where(match >= thresh)
        for i in range(len(temp_y)):
            self._matchedYCoordPreNMS.append(temp_y[i])
            self._matchedXCoordPreNMS.append(temp_x[i])

        self.NonMaxSuppression()

    def visualizeMatchBeforeNMS(self):
        """
        Visualize the match before NMS
        :return: Visualize the match
        """
        clone = self._source.copy()
        print("[INFO] {} matched locations *before* NMS".format(len(self._matchedYCoordPreNMS)))

        for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
            cv2.circle(clone, (int(x + self._intersection[0]),
                               int(y + self._intersection[1])),
                       2, (0, 255, 0), -1)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        cv2.imshow("Before NMS", clone)
        cv2.waitKey(0)

    def NonMaxSuppression(self, dx=2 / 3):
        """
        Reduce the amount of overlapping points in the template matching process
        :param dx: size of the rectangle to check for overlap
        :return: None
        """

        W, H = self._template.shape[:2]
        rects = []

        for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
            rects.append((x, y, x + dx * H, y + dx * W))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        for (startX, startY, endX, endY) in pick:
            x_inter = int(startX + self._intersection[0])
            y_inter = int(startY + self._intersection[1])
            if self._polygon.contains(pl.Point(x_inter, y_inter)):
                self._matchedXCoordPostNMS.append(x_inter)
                self._matchedYCoordPostNMS.append(y_inter)

    def visualizeMatchAfterNonMaxSuppression(self):
        """
        Visualize the match after NMS
        :param dx: size of the rectangle to check for overlap
        :return: None
        """

        clone = self._source.copy()
        for i in range(len(self._matchedXCoordPostNMS)):
            cv2.circle(clone, (self._matchedXCoordPostNMS[i], self._matchedYCoordPostNMS[i]),
                       2, (0, 255, 0), -1)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 800, 600)
        cv2.imshow("After NMS", clone)
        cv2.waitKey(0)

    def matching_displacement(self, target):
        """
        Find the correspondence between two set of intersection points
        on the image
        :param target: target object
        :return: A correspondence between the intersection points
        """
        distance_thresh = 0.25 * self.get_length()

        # Construct arrays of coordinates for each point
        source = np.column_stack((self._matchedXCoordPostNMS,
                                  self._matchedYCoordPostNMS))
        target = np.column_stack((target.get_x_coord(),
                                  target.get_y_coord()))

        correspondence = []
        for x in source:
            dist_x = []
            for y in target:
                dist_x.append(np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2))

            min_index = np.argmin(dist_x)  # Find the minimum distance for the current source point
            if dist_x[min_index] < distance_thresh:
                correspondence.append([x, target[min_index]])

        return correspondence

    def get_x_coord(self):
        return self._matchedXCoordPostNMS

    def get_y_coord(self):
        return self._matchedYCoordPostNMS

    def get_length(self):
        return self._intersection[2]


def main():
    parser = argparse.ArgumentParser(description="Template Matching")
    parser.add_argument("source_image", help="Path to the source image")
    parser.add_argument("template", help="Path to the template image")
    parser.add_argument("intersection", help="Path to the intersection file")
    args = parser.parse_args()

    """
    Sample visualization of the result of template matching on the image
    """
    matcher = TemplateMatcher(args.source_image, args.template, args.intersection)
    matcher.set_boundary()
    matcher.match_template()
    matcher.visualizeMatchBeforeNMS()
    matcher.visualizeMatchAfterNonMaxSuppression()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
