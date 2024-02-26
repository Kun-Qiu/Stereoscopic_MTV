import numpy as np
import cv2
import argparse


class TemplateDrawer:
    def __init__(self, image_path, template_path):
        self.image_path = image_path
        self.template_path = template_path
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.img = cv2.imread(image_path)
        self.clone = self.img.copy()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
                self.ix, self.iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)

    def save_template(self):
        cv2.imwrite(self.template_path, self.img)
        print("Template saved successfully!")

    def draw_template(self):
        cv2.namedWindow("Select Template")
        cv2.setMouseCallback("Select Template", self.draw_line)

        while True:
            cv2.imshow("Select Template", self.img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                self.save_template()
                break
            elif key == ord("r"):
                self.img = self.clone.copy()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image where we'll draw the template")
    ap.add_argument("-t", "--template", type=str, required=True,
                    help="path to save the template image")
    args = vars(ap.parse_args())

    drawer = TemplateDrawer(args["image"], args["template"])
    drawer.draw_template()
