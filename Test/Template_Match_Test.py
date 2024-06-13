import cv2
import TemplateMatching
import matplotlib.pyplot as plt
import numpy as np

# Paths to images
source_path = '../Data/Source/frame_0.png'
target_path = '../Data/Target/frame_0_2us.png'

# Initialize template matchers
template_object = TemplateMatching.TemplateMatcher(source_path, target_path)
template_object.match_template_driver()

# displacement = template_object.get_displacement()
# interpolation = template_object.displacement_interpolation(displacement)
#
# for i in range(0, len(interpolation), 10):
#     for j in range(0, len(interpolation[i]), 10):
#         vector = interpolation[i][j]
#         plt.arrow(i, j, vector[0], vector[1],
#                   head_width=2, head_length=4, fc='red', ec='red')
# plt.title("Approximation of the intersection position")
# plt.axis('off')
# plt.show()
