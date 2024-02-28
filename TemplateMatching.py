import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from Utility.DeNoise import denoised_image

# Load the images
img = cv2.imread('Data/Source/source.png')
img_gray = denoised_image('Data/Source/source.png')
temp = cv2.imread('Data/Template/temp.jpg')
temp_gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

# Apply Tsutomu's thresholding
_, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, temp_thresh = cv2.threshold(temp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow("img", img_thresh)
cv2.imshow("temp", cv2.resize(temp_thresh, (520, 520)))
cv2.waitKey()

# save the image dimensions
W, H = temp.shape[:2]

# Passing the image to matchTemplate method
match = cv2.matchTemplate(
    image=img_thresh, templ=temp_thresh,
    method=cv2.TM_CCOEFF_NORMED)

# Define a minimum threshold
thresh = 0.6

# Select rectangles with confidence greater than threshold
(y_points, x_points) = np.where(match >= thresh)

# initialize our list of rectangles
boxes = list()

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(x_points, y_points):
    # update our list of rectangles
    boxes.append((x, y, x + W, y + H))

# apply non-maxima suppression to the rectangles
# this will create a single bounding box
boxes = non_max_suppression(np.array(boxes))

# Define scaling factors
scale_width = 0.1  # Scale width by 80%
scale_height = 0.1  # Scale height by 80%

# loop over the final bounding boxes
for (x1, y1, x2, y2) in boxes:
    # Compute the scaled width and height
    scaled_width = int((x2 - x1) * scale_width)
    scaled_height = int((y2 - y1) * scale_height)
    # Compute new coordinates for the bounding box
    new_x2 = x1 + scaled_width
    new_y2 = y1 + scaled_height
    # draw the scaled bounding box on the image
    cv2.rectangle(img, (x1, y1), (new_x2, new_y2), (0, 255, 0), 3)

# Show the template and the final output
cv2.imshow("After NMS", img)
cv2.waitKey(0)

# destroy all the windows
# manually to be on the safe side
cv2.destroyAllWindows()
