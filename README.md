# Project 2D Image to Stereoscopic Views

This code project a 2D image onto two image planes for stereoscopic image viewing.

## Dependencies
- `cv2`: OpenCV library for image processing
- `numpy`: Numerical computing library
- `torch`: PyTorch deep learning library
- `torch.optim`: PyTorch optimization module
- `OpticalFlow`: Custom module for optical flow computation
- `TemplateMatching`: Custom module for template matching
- `matplotlib.pyplot`: Plotting library

## Utility Functions

1. `translateImage(image, translateField)`: This function applies a displacement field to an input image, effectively translating the image pixels.

2. `correspondence_displacement(correspondence_list)`: This function takes a list of correspondences between initial and final points, and computes the displacement (dx, dy) for each correspondence.

3. `visualize_displacement(image, name, field)`: This function visualizes the displacement vectors on top of the original image.

4. `visualize_displacement_difference(field1, field2, image)`: This function visualizes the difference between two displacement fields overlayed on the original image.

## Displacement Field Model

The `DisplacementFieldModel` class is a PyTorch module that represents the displacement field as learnable parameters.

## Loss Functions

The code defines several loss functions to optimize the displacement field:

1. `smoothness_constraint(u, v)`: This function computes a smoothness constraint loss based on the gradients of the displacement field.

2. `intensity_constraint(source_img, target_img, predicted_field, lambda_intensity)`: This function computes the intensity difference loss between the translated source image and the target image.

3. `known_displace_constraint(optical_flow, template_flow, lambda_vel)`: This function computes the loss based on the difference between the predicted displacement field and the known displacement field.

## Optimization

The `optimize_displacement_field` function performs the main training loop to optimize the displacement field model by minimizing the combined loss.

## Driver Code

The driver code loads the source, target, and template images, and performs optical flow and template matching to obtain the initial displacement field. It then initializes the displacement field model and optimizes it using the defined loss functions. Finally, it visualizes the initial and optimized displacement fields.

To run this code, you will need to provide the necessary input files (`source_path`, `target_path`, `template_path`, and `intersection`). The code will then generate the stereoscopic views by optimizing the displacement field.