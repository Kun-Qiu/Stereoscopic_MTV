import cv2
import TemplateMatching
import matplotlib.pyplot as plt
import numpy as np

# Paths to images
source_path = '../Data/Source/source_25.png'
target_path = '../Data/Target/target_25.png'

# Initialize template matchers
template_object = TemplateMatching.TemplateMatcher(source_path, target_path)
template_object.match_template_driver()