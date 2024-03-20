import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import matplotlib.pyplot as plt


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
        self._scaledIntersection = []

    def match_template(self, scale_num=3):
        """
        Main driver for template matching using openCV
        :return: None
        """
        gray_source = self._source
        if len(self._source.shape) == 3:  # Check if the source image is color (not grayscale)
            gray_source = cv2.cvtColor(self._source, cv2.COLOR_BGR2GRAY)

        _, img_thresh = cv2.threshold(gray_source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_gray = cv2.cvtColor(self._template, cv2.COLOR_BGR2GRAY)

        for iteration in range(scale_num):
            scale = 1 - 0.1 * iteration

            new_height = int((1 - scale) * temp_gray.shape[0] / 2)
            new_width = int((1 - scale) * temp_gray.shape[1] / 2)

            # Crop the image from both sides
            scaled_template = temp_gray[new_height:(temp_gray.shape[0] - new_height),
                              new_width:(temp_gray.shape[1] - new_width)]

            match = cv2.matchTemplate(image=img_thresh,
                                      templ=scaled_template,
                                      method=cv2.TM_CCOEFF_NORMED)

            intersection_point = [self._intersection[0] - new_width,
                                  self._intersection[1] - new_height]

            # Minimum correlation threshold
            thresh = 0.6
            temp_y, temp_x = np.where(match >= thresh)
            for i in range(len(temp_y)):
                self._matchedYCoordPreNMS.append(temp_y[i])
                self._matchedXCoordPreNMS.append(temp_x[i])
                self._scaledIntersection.append(intersection_point)

        self.NonMaxSuppression()

    def visualizeMatchBeforeNMS(self):
        """
        Visualize the match before NMS
        :return: Visualize the match
        """
        clone = self._source.copy()
        print("[INFO] {} matched locations *before* NMS".format(len(self._matchedYCoordPreNMS)))

        count = 0
        for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
            cv2.circle(clone, (int(x + self._scaledIntersection[count][0]),
                               int(y + self._scaledIntersection[count][1])),
                       2, (0, 255, 0), -1)
            count += 1
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
            rects.append((x, y, x + dx * W, y + dx * H))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        count = 0
        for (startX, startY, endX, endY) in pick:
            x_inter = int(startX + self._scaledIntersection[count][0])
            y_inter = int(startY + self._scaledIntersection[count][1])
            self._matchedXCoordPostNMS.append(x_inter)
            self._matchedYCoordPostNMS.append(y_inter)

            count += 1

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
        cv2.imshow("After NMS", clone)
        cv2.waitKey(0)

    def matching_displacement(self, target):
        """
        Find the correspondence between two set of intersection points
        on the image
        :param target: target object
        :return: A correspondence between the intersection points
        """
        distance_thresh = 0.5 * self._intersection[2]

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
    matcher.match_template()
    matcher.visualizeMatchBeforeNMS()
    matcher.visualizeMatchAfterNonMaxSuppression()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
