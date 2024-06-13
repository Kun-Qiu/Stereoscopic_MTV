import cv2
import numpy as np
import argparse

"""
ImageLineDrawer is a class use to determine the 
template image for the template matching
"""

"""
Static Utility Function
"""


def obtain_line(img, pt1, pt2, color=(0, 255, 0), thickness=2):
    """
    Using the OpenCV line function, a line is drawn on the input image
    :param img: image
    :param pt1: first point on the line
    :param pt2: second point on the line
    :param color: color of the line
    :param thickness: thickness of the line
    :return: draw the line on the input img
    """
    return cv2.line(img, pt1, pt2, color, thickness)


class Template:
    def __init__(self, image_path, save_path):
        self.image = cv2.imread(image_path)
        self.save_path = save_path
        self.points = []

    def user_prompted_points(self):
        """
        Allows the user to select the intersection points on the source image to
        extract a template for the template matching algorithm
        :return: None
        """
        print("Please pick four points on the image by clicking on them. Press 'q' to quit.")

        source_img_copy = self.image.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(source_img_copy, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Template", source_img_copy)
                if len(self.points) == 4:
                    cv2.destroyAllWindows()
                print(x, y)

        cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Template", 600, 600)
        cv2.imshow('Template', source_img_copy)
        cv2.setMouseCallback('Template', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_lines_on_image(self):
        """
        Ask the user to choose one intersection to be selected as the template, draw
        the lines in the image
        :return: None
        """
        self.user_prompted_points()
        copied_image = self.image.copy()

        obtain_line(copied_image,
                    self.points[0],
                    self.points[1])

        obtain_line(copied_image,
                    self.points[2],
                    self.points[3])

        cv2.namedWindow("Line Template", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Line Template", 600, 600)
        cv2.imshow("Line Template", copied_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_lines_image(self):
        """
        Draw the lines on the image, crop the desired section for template,
        save the intersection point.
        :return: [Template, Intersection Position]
        """

        min_x = min(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        max_x = max(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        min_y = min(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])
        max_y = max(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])

        # length to adjacent point
        length = 0.5 * np.sqrt((self.points[0][0] - self.points[1][0]) ** 2 +
                               (self.points[0][1] - self.points[1][1]) ** 2)

        # Crop the image to the bounding box
        cropped_image = self.image[min_y:max_y, min_x:max_x]
        cv2.imwrite(self.save_path, cropped_image)

        with open('length.txt', 'w') as f:
            f.write(length)

    def find_intersection(self):
        """
        Given four points, determine the intersection point
        :return: x, y coordinate of intersection point
        """
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        x3, y3 = self.points[2]
        x4, y4 = self.points[3]

        # Calculate intersection point
        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
                         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
                         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        # return int(intersection_x), int(intersection_y)
        return intersection_x, intersection_y

    def run(self):
        """
        Run the necessary function to generate the template and the
        intersection points
        :return: [Template, Intersection Points]
        """
        self.draw_lines_on_image()
        return self.save_lines_image()


if __name__ == "__main__":
    """
    Argument parsing --> Driver code 
    Example terminal command -> 
    python \Template.py path_to_image path_to template
    """
    parser = argparse.ArgumentParser(description='Draw lines on an image')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('save_path', type=str, help='Path to save image')
    args = parser.parse_args()

    image_drawer = Template(args.image_path, args.save_path)
    image_drawer.run()
