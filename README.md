# 3D Velocity Field Approximation Using Stereoscopic Vision

This project enables the extraction of 3D velocity field from two sets of images from a stereoscopic setup using 
optimization of the project of a predicted 3D field onto the two stereoscopic views.

## Dependencies
- `cv2`: OpenCV library for image processing
- `numpy`: Numerical computing library
- `torch`: PyTorch deep learning library

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