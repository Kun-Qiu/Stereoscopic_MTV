import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse


class TemplateMatcher:
    def __init__(self, source_image_path, template_path, intersection_path):
        """
        Class initialization for the template matcher
        :param source_image_path: path to source image
        :param template_path: path to template image
        :param intersection_path: path to intersection txt
        """

        self.source = cv2.imread(source_image_path)
        self.template = cv2.imread(template_path)
        self.intersection = np.loadtxt(intersection_path)
        self.matchedXCoordPreNMS = []
        self.matchedYCoordPreNMS = []
        self.matchedXCoordPostNMS = []
        self.matchedYCoordPostNMS = []

    def match_template(self):
        """
        Main driver for template matching using openCV
        :return: None
        """
        # Apply Tsutomu's thresholding
        _, img_thresh = cv2.threshold(self.source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        temp_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        match = cv2.matchTemplate(
            image=img_thresh, templ=temp_gray,
            method=cv2.TM_CCOEFF_NORMED)

        # Minimum correlation threshold
        thresh = 0.6
        (self.matchedYCoordPreNMS, self.matchedXCoordPreNMS) = np.where(match >= thresh)

    def visualizeMatchBeforeNMS(self):
        """
        Visualize the match before NMS
        :return: Visualize the match
        """
        W, H = self.template.shape[:2]
        clone = self.source.copy()
        print("[INFO] {} matched locations *before* NMS".format(len(self.matchedYCoordPreNMS)))

        for (x, y) in zip(self.matchedXCoordPreNMS, self.matchedYCoordPreNMS):
            cv2.rectangle(clone, (x, y), (x + W, y + H), (0, 255, 0), 3)
        cv2.imshow("Before NMS", clone)
        cv2.waitKey(0)

    def VisualizeMatchAfterNonMaxSuppression(self, dx=0.4):
        """
        Visualize the match after NMS
        :param dx: size of the rectangle to check for overlap
        :return: Visualize the match
        """

        W, H = self.template.shape[:2]
        clone = self.source.copy()
        rects = []

        for (x, y) in zip(self.matchedXCoordPreNMS, self.matchedYCoordPreNMS):
            rects.append((x, y, x + dx * W, y + dx * H))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        for (startX, startY, endX, endY) in pick:
            cv2.circle(clone, (int(startX + self.intersection[0]), int(startY + self.intersection[1])),
                       2, (0, 255, 0), -1)
            self.matchedXCoordPostNMS.append(startX)
            self.matchedYCoordPostNMS.append(startY)
        cv2.imshow("After NMS", clone)
        cv2.waitKey(0)

    def get_x_coord(self):
        return self.matchedXCoordPostNMS

    def get_y_coord(self):
        return self.matchedYCoordPostNMS

    # def displacement(self, other):


def main():
    parser = argparse.ArgumentParser(description="Template Matching")
    parser.add_argument("source_image", help="Path to the source image")
    parser.add_argument("template", help="Path to the template image")
    parser.add_argument("intersection", help="Path to the intersection file")
    args = parser.parse_args()

    matcher = TemplateMatcher(args.source_image, args.template, args.intersection)
    matcher.match_template()
    matcher.visualizeMatchBeforeNMS()
    matcher.VisualizeMatchAfterNonMaxSuppression()
