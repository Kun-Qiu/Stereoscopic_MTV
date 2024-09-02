import cv2
import OpticalFlow
import matplotlib.pyplot as plt
from Utility import Visualization as vis

right_initial_path = "../Tool_3D/Velocity Test/Right_(0,0,0).png"
right_final_path = "../Tool_3D/Velocity Test/Right_(2,3,0)_0mm.png"
left_initial_path = "../Tool_3D/Velocity Test/Left_(0,0,0).png"
left_final_path = "../Tool_3D/Velocity Test/Left(2,3,0)_0mm.png"

right_initial_img = cv2.imread(right_initial_path)
right_final_img = cv2.imread(right_final_path)
left_initial_img = cv2.imread(left_initial_path)
left_final_img = cv2.imread(left_final_path)

# Optical Flow
of_object_right = OpticalFlow.OpticalFlow(right_initial_path, right_final_path)
of_object_left = OpticalFlow.OpticalFlow(left_initial_path, left_final_path)

of_object_right.calculate_optical_flow()
of_flow_right = of_object_right.get_flow()

of_object_left.calculate_optical_flow()
of_flow_left = of_object_left.get_flow()

vis.visualize_displacement(right_initial_img, "Initial Displacement", of_flow_right, 5)
vis.visualize_displacement(left_initial_img, "Left Displacement", of_flow_left, 5)

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
# plt.title('Source Image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
# plt.title('Target Image')
# plt.axis('off')

# plt.tight_layout()
plt.show()
