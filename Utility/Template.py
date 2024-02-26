import cv2
import argparse


class ImageLineDrawer:
    def __init__(self, image_path, size_multiplier):
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (0, 0), fx=size_multiplier, fy=size_multiplier)
        self.points = []
        self.size_multiplier = size_multiplier

    def draw_line(self, pt1, pt2, color=(0, 255, 0), thickness=1):
        return cv2.line(self.image, pt1, pt2, color, thickness)

    def get_points(self):
        print("Please pick two points on the image by clicking on them. Press 'q' to quit.")

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                if len(self.points) == 2:
                    cv2.destroyAllWindows()

        cv2.imshow('Image', self.image)
        cv2.setMouseCallback('Image', mouse_callback)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def draw_lines(self):
        self.get_points()
        self.draw_line(self.points[0], self.points[1])
        self.points.clear()  # Clear the points list after drawing the first line

        self.get_points()
        final_image_with_second_line = self.draw_line(self.points[0], self.points[1])
        cv2.imshow('Final Image', final_image_with_second_line)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return final_image_with_second_line

    def save_lines_image(self, image_path):
        lines_image = self.draw_lines()
        self.image = cv2.resize(self.image, (0, 0), fx=1/self.size_multiplier,
                                fy=1/self.size_multiplier)
        cv2.imwrite(image_path, lines_image)
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
