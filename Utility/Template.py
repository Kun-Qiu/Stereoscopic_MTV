import cv2
import numpy as np
import argparse


class ImageLineDrawer:
    def __init__(self, image_path, size_multiplier):
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (0, 0), fx=size_multiplier, fy=size_multiplier)
        self.points = []
        self.size_multiplier = size_multiplier
        self.size = np.shape(self.image)

    def draw_line(self, pt1, pt2, color=(0, 255, 0), thickness=1):
        return cv2.line(self.image, pt1, pt2, color, thickness)

    def get_points(self):
        print("Please pick two points on the image by clicking on them. Press 'q' to quit.")

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                if len(self.points) == 4:
                    cv2.destroyAllWindows()
                print(x, y)

        cv2.imshow('Image', self.image)
        cv2.setMouseCallback('Image', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_lines(self):
        self.get_points()
        self.draw_line(self.points[0], self.points[1])
        self.draw_line(self.points[2], self.points[3])

        cv2.imshow('Final Image', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.image

    def save_lines_image(self, image_path):
        self.draw_lines()

        # Create a blank image with the calculated dimensions
        lines_image = np.zeros(np.shape(self.image), dtype=np.uint8)

        # Draw the lines on the blank image
        cv2.line(lines_image, (self.points[0][0], self.points[0][1]),
                 (self.points[1][0], self.points[1][1]), (0, 255, 0), 1)
        cv2.line(lines_image, (self.points[2][0], self.points[2][1]),
                 (self.points[3][0], self.points[3][1]), (0, 255, 0), 1)

        cv2.resize(lines_image, (self.image.shape[1], self.image.shape[0]))

        min_x = min(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        max_x = max(self.points[0][0], self.points[1][0], self.points[2][0], self.points[3][0])
        min_y = min(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])
        max_y = max(self.points[0][1], self.points[1][1], self.points[2][1], self.points[3][1])

        # Crop the image to the bounding box
        cropped_image = lines_image[min_y:max_y, min_x:max_x]

        # Save the lines as an image
        cv2.imwrite(image_path, cropped_image)
        print(f"Image with only lines drawn saved as {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw lines on an image')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('save_path', type=str, help='Path to save image')
    parser.add_argument('--multi', type=float, default=2,
                        help='Multiplier to resize the image (default: 2)')
    args = parser.parse_args()

    image_drawer = ImageLineDrawer(args.image_path, args.multi)
    image_drawer.save_lines_image(args.save_path)
