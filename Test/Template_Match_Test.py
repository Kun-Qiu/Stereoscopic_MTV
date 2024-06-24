import cv2
import TemplateMatching
import matplotlib.pyplot as plt
import numpy as np
import Utility.TifReader as tr
import os
from Utility import Template as tp

# tif = tr.TifReader('../Data/Source/bp0_0p0_0microsec.tif',
#                    '../Data/Source/source_avg.png')
# source = tif.average_tif(save_path=True)

# Paths to images
# source_path = '../Data/Source/source_avg.png'
# target_path = '../Data/Synthetic Target/img1.png'
# template_path = '../Data/Template/frame_0_temp.png'

# source_path = '../Data/Source/source_25.png'
# target_path = '../Data/Target/target_25.png'
# template_path = '../Data/Template/template25.png'

# source_path = '../Data/Source/source_25_new.png'
# target_path = '../Data/Target/target_25_new.png'
# template_path = '../Data/Template/template25.png'

# source_path = '../Data/Source/frame_2.png'
# target_path = '../Data/Target/frame_2_2us.png'
# template_path = '../Data/Template/frame_2_temp.png'

experiment_path = "../2D_Experiment/Experiment"
snr_values = [1, 2, 4, 8, 16]
set_range = 1
thresh_val = [0.83, 0.75, 0.7]

for value in snr_values:
    snr_path = os.path.join(experiment_path, f"SNR_{value}")
    for i in range(set_range):
        for thresh in thresh_val:
            set_path = os.path.join(snr_path, f"Set_{i}")
            source_path = os.path.join(set_path, f"Gaussian_Grid_Image_Set_{i}.png")
            template_path = os.path.join(snr_path, f"Template.png")

            if not os.path.exists(template_path):
                template = tp.Template(source_path, template_path)
                template.run()

            translate_path = os.path.join(set_path, f"Rotational_Flow_Image_Set_{i}.png")

            # Initialize template matchers
            template_object = TemplateMatching.TemplateMatcher(source_path, translate_path, template_path,
                                                               thresh_source=thresh,
                                                               thresh_target=thresh)
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
