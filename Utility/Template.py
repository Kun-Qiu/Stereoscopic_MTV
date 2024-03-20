import cv2
import numpy as np
import argparse

"""
ImageLineDrawer is a class use to determine the 
template image for the template matching
"""


class ImageLineDrawer:
    def __init__(self, image_path, size_multiplier):
        self.image = cv2.imread(image_path)
        self.resized_image = cv2.resize(self.image, (0, 0), fx=size_multiplier, fy=size_multiplier)
        self.points = []
        self.multiplier = size_multiplier

    def draw_line(self, img, pt1, pt2, color=(0, 255, 0), thickness=4):
        """
        :param img: image
        :param pt1: first point on the line
        :param pt2: second point on the line
        :param color: color of the line
        :param thickness: thickness of the line
        :return: draw the line on the input img
        """
        return cv2.line(img, pt1, pt2, color, thickness)

    def get_points(self):
        """
        Function allows the user to select points on the image
        :return: None
        """
        print("Please pick two points on the image by clicking on them. Press 'q' to quit.")

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                if len(self.points) == 4:
                    cv2.destroyAllWindows()
                print(x, y)

        cv2.imshow('Image', self.resized_image)
        cv2.setMouseCallback('Image', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_lines(self):
        """
        Lines are drawn on the image using the points obtained
        in mouse click event
        :return: None
        """
        self.get_points()
        copied_image = self.resized_image.copy()
        self.draw_line(copied_image,
                       self.points[0],
                       self.points[1])

        self.draw_line(copied_image,
                       self.points[2],
                       self.points[3])

        cv2.imshow('Final Image', copied_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_lines_image(self, image_path):
        """
        Draw the lines on the image, crop the desired section for template,
        save the intersection point.
        :param image_path: Path which image will be saved
        :return: None
        """
        self.draw_lines()

        gray_img = cv2.cvtColor(self.resized_image, cv2.COLOR_RGB2GRAY)
        _, img_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        min_x = min(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        max_x = max(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        min_y = min(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])
        max_y = max(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])

        # length to adjacent point
        length = 0.5 * np.sqrt((self.points[0][0] - self.points[1][0]) ** 2 +
                               (self.points[0][1] - self.points[1][1]) ** 2)

        inter_x, inter_y = self.find_intersection()
        intersection_point_relative = np.array([int((inter_x - min_x) / self.multiplier),
                                                int((inter_y - min_y) / self.multiplier),
                                                int(length)])

        # Crop the image to the bounding box
        cropped_image = cv2.resize(img_thresh[min_y:max_y, min_x:max_x],
                                   (0, 0), fx=1 / self.multiplier, fy=1 / self.multiplier)
        # Writing the template to the Template Folder
        cv2.imwrite(image_path, cropped_image)
        np.savetxt('Data/Template/intersection.txt', intersection_point_relative)
        print(f"Region of the image overlapping with lines saved as {image_path}")

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
        return int(intersection_x), int(intersection_y)


if __name__ == "__main__":
    """
    Argument parsing --> Driver code 
    Example terminal command -> 
    python Template.py path_to_image path_to template --multi image_scale
    """
    parser = argparse.ArgumentParser(description='Draw lines on an image')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('save_path', type=str, help='Path to save image')
    parser.add_argument('--multi', type=float, default=2,
                        help='Multiplier to resize the image (default: 2)')
    args = parser.parse_args()

    image_drawer = ImageLineDrawer(args.image_path, args.multi)
    image_drawer.save_lines_image(args.save_path)
