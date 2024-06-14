import cv2
import TemplateMatching
import matplotlib.pyplot as plt
import numpy as np
import Utility.TifReader as tr

# tif = tr.TifReader('../Data/Source/bp0_0p0_0microsec.tif',
#                    '../Data/Source/source_avg.png')
# source = tif.average_tif(save_path=True)

# Paths to images
source_path = '../Data/Source/source_avg.png'
target_path = '../Data/Synthetic Target/img1.png'
template_path = '../Data/Template/frame_0_temp.png'

# source_path = '../Data/Source/source_25.png'
# target_path = '../Data/Target/target_25.png'
# template_path = '../Data/Template/template25.png'

# source_path = '../Data/Source/source_25_new.png'
# target_path = '../Data/Target/target_25_new.png'
# template_path = '../Data/Template/template25.png'

# source_path = '../Data/Source/frame_2.png'
# target_path = '../Data/Target/frame_2_2us.png'
# template_path = '../Data/Template/frame_2_temp.png'

# Initialize template matchers
template_object = TemplateMatching.TemplateMatcher(source_path, target_path, template_path)
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
