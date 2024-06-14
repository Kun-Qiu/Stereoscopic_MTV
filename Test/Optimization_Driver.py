import cv2
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt
from Utility import Visualization as vis
import Optimization as op
from Optimization import optimize_displacement_field
import torch


# Load original image and displacement field (example)
source_path = '../Data/Source/source_avg.png'
target_path = '../Data/Synthetic Target/synthetic_0.png'
template_path = '../Data/Template/frame_0_temp.png'

# Optical Flow
of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()

# Template Matching
template_object = TemplateMatching.TemplateMatcher(source_path, target_path, template_path)
template_object.match_template_driver()

predicted = of_object.get_flow()
observed = template_object.get_displacement()

# ----------------------- Optimization ---------------------------------------------------------------
# Initialize displacement field model and optimizer
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

model = op.DisplacementFieldModel(predicted)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
optimized_displacement = optimize_displacement_field(model, source_image_tensor,
                                                     target_image_tensor,
                                                     observed, optimizer,
                                                     50, 300, 10000)

vis.visualize_displacement(source_image, "Optimized Displacement", optimized_displacement)
vis.visualize_displacement(source_image, "Initial Displacement", predicted)
vis.visualize_displacement_difference(optimized_displacement, predicted, source_image)
plt.show()
