import torch
import torch.nn.functional as F
import torch.optim as optim
import OpticalFlow
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Utility Functions
def displace_image(image, displacement_field, tensor=False):
    """
    Displace the source image by the displacement field pixel by pixel
    :param image: source image
    :param displacement_field: displacement field
    :param tensor: whether to set return as tensor
    :return: Displaced image (tensor or numpy.ndarray)
    """
    height, width = image.shape[:2]
    displaced_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            displacement_x, displacement_y = displacement_field[y, x]

            new_x = int(x + displacement_x)
            new_y = int(y + displacement_y)
            if 0 <= new_x < width and 0 <= new_y < height:
                displaced_image[y, x] = image[new_y, new_x]

    if tensor:
        return torch.tensor(displaced_image, dtype=torch.float32)

    return displaced_image


# def known_displacement_loss(predicted_displacement, known_displacement):
#


# __________________________________________________________________________________________
class DisplacementFieldModel(torch.nn.Module):
    def __init__(self, initial_guess):
        super(DisplacementFieldModel, self).__init__()

        self.displacements = torch.nn.Parameter(initial_guess.clone().detach())

    def forward(self):
        return self.displacements


def displacement_loss(source, target, predicted_field, target_field=None, lambda_vel=0.1, lambda_tv=0.1):
    """
    Calculation of the loss function of the displacement field with 3 Components:
    Intensity Difference (Optical Flow), Velocity Difference (Template Matching),
    and total variation in intensity
    :param source: source image
    :param target: target image
    :param predicted_field: predicted displacement field
    :param target_field: template matching displacement field
    :param lambda_vel: velocity regularizer
    :param lambda_tv: variation regularizer
    :return: The total loss
    """
    # Component 1: Difference between source image after applying displacement field and target image
    predicted_image = displace_image(source, predicted_field, tensor=True)
    loss_component1 = torch.mean(torch.square(predicted_image - target))

    # # Component 2: Difference between known displacement at certain locations
    # loss_component2 = lambda_tv * torch.mean(torch.square(predicted_displacements - observed_displacements))

    # # Component 3: Total variation regularization
    # loss_component3 = total_variation_regularization(predicted_displacements)
    #
    # # Combine components with appropriate weights
    total_loss = loss_component1
    return total_loss


# Function to compute total variation regularization
def total_variation_regularization(displacement_field):
    tv_loss = torch.sum(torch.abs(displacement_field[:, :, :, :-1] - displacement_field[:, :, :, 1:])) + \
              torch.sum(torch.abs(displacement_field[:, :, :-1, :] - displacement_field[:, :, 1:, :]))
    return tv_loss


# Step 3: Optimization loop
def optimize_displacement_field(model, observed_displacements, optimizer, num_epochs=1000):
    for epoch in range(num_epochs):
        # Forward pass: compute predicted displacements
        predicted_displacements = model()

        # Compute the loss
        loss = displacement_loss(predicted_displacements, observed_displacements)

        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# Load original image and displacement field (example)
source_path = 'Data/Source/source.png'
target_path = 'Data/Target/frame_31.png'
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()

predicted_target_image = displace_image(source_image, of_object.get_flow())
loss = displacement_loss(source_image, target_image, of_object.get_flow())
print(loss)

# Display the image
plt.imshow(predicted_target_image)
plt.axis('off')
plt.show()

# # Assuming grid_images are your input images
# grid_images = ...  # Provide your grid images here
# observed_displacements = ...  # Provide your observed displacements here
#
# # Initialize your displacement field model and optimizer
# model = DisplacementFieldModel()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # Optimize the displacement field
# optimize_displacement_field(model, observed_displacements, optimizer)
