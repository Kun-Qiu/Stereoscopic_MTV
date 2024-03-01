import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from Utility.DeNoise import denoised_image

# Load the images
img = cv2.imread('Data/Source/source.png')
img_gray = denoised_image('Data/Source/source.png')

img2 = cv2.imread('Data/Target/target.png')
img_target = denoised_image('Data/Target/target.png')
temp = cv2.imread('Data/Template/temp.jpg')
temp_gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

intersection = np.loadtxt('Data/Template/intersection.txt')

# Apply Tsutomu's thresholding
_, img_thresh = cv2.threshold(img_target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# save the image dimensions
W, H = temp.shape[:2]

# Passing the image to matchTemplate method
match = cv2.matchTemplate(
    image=img_thresh, templ=temp_gray,
    method=cv2.TM_CCOEFF_NORMED)

# Define a minimum threshold
thresh = 0.6

# Select rectangles with confidence greater than threshold
(y_points, x_points) = np.where(match >= thresh)
clone = img.copy()
print("[INFO] {} matched locations *before* NMS".format(len(y_points)))

for (x, y) in zip(x_points, y_points):
    # draw the bounding box on the image
    cv2.rectangle(clone, (x, y), (x + W, y + H), (0, 255, 0), 3)
cv2.imshow("Before NMS", clone)

clone2 = img2.copy()
dx = 0.4
rects = []

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(x_points, y_points):
    # update our list of rectangles
    rects.append((x, y, x + dx * W, y + dx * H))
# apply non-maxima suppression to the rectangles

pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))
# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
    # draw the bounding box on the image
    # cv2.rectangle(clone2, (startX, startY), (endX, endY),
    #               (255, 0, 0), 3)
    cv2.circle(clone2, (startX + int(intersection[0]), startY + int(intersection[1])), 2, (0, 255, 0), -1)
# show the output image
cv2.imshow("After NMS", clone2)
cv2.waitKey(0)
