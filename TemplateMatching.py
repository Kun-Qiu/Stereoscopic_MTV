import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
from scipy.spatial.distance import cdist


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

    def match_template(self):
        """
        Main driver for template matching using openCV
        :return: None
        """
        gray_source = self._source
        if len(self._source.shape) == 3:  # Check if the source image is color (not grayscale)
            gray_source = cv2.cvtColor(self._source, cv2.COLOR_BGR2GRAY)

        _, img_thresh = cv2.threshold(gray_source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_gray = cv2.cvtColor(self._template, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(
            image=img_thresh, templ=temp_gray,
            method=cv2.TM_CCOEFF_NORMED)

        # Minimum correlation threshold
        thresh = 0.6
        (self._matchedYCoordPreNMS, self._matchedXCoordPreNMS) = np.where(match >= thresh)

    def visualizeMatchBeforeNMS(self):
        """
        Visualize the match before NMS
        :return: Visualize the match
        """
        W, H = self._template.shape[:2]
        clone = self._source.copy()
        print("[INFO] {} matched locations *before* NMS".format(len(self._matchedYCoordPreNMS)))

        for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
            cv2.rectangle(clone, (x, y), (x + W, y + H), (0, 255, 0), 3)
        cv2.imshow("Before NMS", clone)
        cv2.waitKey(0)

    def visualizeMatchAfterNonMaxSuppression(self, dx=0.4):
        """
        Visualize the match after NMS
        :param dx: size of the rectangle to check for overlap
        :return: Visualize the match
        """

        W, H = self._template.shape[:2]
        clone = self._source.copy()
        rects = []

        for (x, y) in zip(self._matchedXCoordPreNMS, self._matchedYCoordPreNMS):
            rects.append((x, y, x + dx * W, y + dx * H))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        for (startX, startY, endX, endY) in pick:
            cv2.circle(clone, (int(startX + self._intersection[0]), int(startY + self._intersection[1])),
                       2, (0, 255, 0), -1)
            self._matchedXCoordPostNMS.append(startX)
            self._matchedYCoordPostNMS.append(startY)
        cv2.imshow("After NMS", clone)
        cv2.waitKey(0)

    def matching_displacement(self, target):
        """
        Find the correspondence between two set of intersection points
        on the image
        :param target: target object
        :return: A correspondence between the intersection points
        """
        x1 = np.array(self._matchedXCoordPostNMS)
        y1 = np.array(self._matchedYCoordPostNMS)
        x2 = np.array(target.get_x_coord)
        y2 = np.array(target.get_y_coord)

        # Construct arrays of coordinates for each point
        points1 = np.column_stack((x1, y1))
        points2 = np.column_stack((x2, y2))

        # Calculate pairwise distances between points
        distances = cdist(points1, points2)

        # Find the indices of the closest point in the second array for each point in the first array
        closest_indices = np.argmin(distances, axis=1)

        correspondences = []

        W, H = self._template.shape[:2]
        thresh = np.sqrt(W ** 2 + H ** 2) / 2

        for i, closest_idx in enumerate(closest_indices):
            # Only consider correspondences where the distance is below a threshold
            if distances[i, closest_idx] < thresh:
                correspondences.append((i, closest_idx))
        return correspondences

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
