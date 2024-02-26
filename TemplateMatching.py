import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import argparse


def select_template(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Display the image and allow the user to select a ROI
    roi = cv2.selectROI("Select Template", image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Crop the selected ROI
    template = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    return template


def main(image_path, threshold):
    # Load the input image
    image = cv2.imread(image_path)

    # Select the template manually
    template = select_template(image_path)
    (tH, tW) = template.shape[:2]

    # Convert both the image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    print("[INFO] performing template matching...")
    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

    # Find all locations in the result map where the matched value is greater than the threshold
    loc = np.where(result >= threshold)

    # Loop over all match locations and draw dots at the center
    for pt in zip(*loc[::-1]):
        # Calculate the center of the match
        center = (pt[0] + int(tW / 2), pt[1] + int(tH / 2))
        # Draw a dot at the center
        cv2.circle(image, center, 5, (255, 0, 0), -1)

    # Apply non-maxima suppression to the matched locations
    pick = non_max_suppression(np.array([center + (center[0] + tW, center[1] + tH) for center in zip(*loc[::-1])]))

    # Loop over the final bounding boxes
    for (startX, startY, endX, endY) in pick:
        # Draw the bounding box on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

    # Show the output image
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image where we'll apply template matching")
    ap.add_argument("-b", "--threshold", type=float, default=0.8,
                    help="threshold for multi-template matching")
    args = vars(ap.parse_args())

    main(args["image"], args["threshold"])
