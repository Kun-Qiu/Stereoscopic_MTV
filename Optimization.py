import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt


# Utility Functions
def displace_image(image, field, tensor=False):
    """
    Displace the source image by the displacement field pixel by pixel
    :param image: source image
    :param field: displacement field
    :param tensor: whether to set return as tensor
    :return: Displaced image (tensor or numpy.ndarray)
    """
    height, width = image.shape[:2]
    displaced_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            displacement_x, displacement_y = field[x, y]

            new_x = int(x + displacement_x)
            new_y = int(y + displacement_y)
            if 0 <= new_x < width and 0 <= new_y < height:
                displaced_image[x, y] = image[new_x, new_y]

    if tensor:
        return torch.tensor(displaced_image, dtype=torch.float32)

    return displaced_image


def correspondence_displacement(correspondence_list):
    """
    Given a corresponding array that match one point to another, determines
    the displacement dx and dy from the initial point
    :param correspondence_list: List containing correspondence between two initial and final position
    :return: A list that contain the dx and dy
    """
    displacement_list = []
    for match in correspondence_list:
        initial_point = match[0]
        final_point = match[1]

        dx = final_point[0] - initial_point[0]
        dy = final_point[1] - initial_point[1]

        displacement_list.append([initial_point[0], initial_point[1], dx, dy])

    return displacement_list


def visualize_displacement(image, field):
    """
    Visualize the displacement vectors on top of the original image
    :param image: Original image
    :param field: Displacement field
    """
    # Calculate magnitude of displacement vectors
    displacement_magnitude = np.linalg.norm(field, axis=-1)

    # Create quiver plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.quiver(field[..., 0], field[..., 1], scale=10)
    plt.imshow(displacement_magnitude, cmap='viridis', alpha=0.5)
    plt.colorbar(label='Magnitude of Displacement')

    plt.title('Displacement Field Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# __________________________________________________________________________________________

### Linear Model that takes in a matrix of shape feature X height X width
### Essentially performing f(Img) = img * A where A is how you want to transform img
### By doing backprogation, we want to find A such that img * A gives you the field 
### That you want. 
class DisplacementFieldModel(torch.nn.Module):
    def __init__(self):
        super(DisplacementFieldModel, self).__init__()
        self.l1 = nn.Linear(2,2)
        # initial_guess = torch.tensor(initial_guess, dtype=torch.float32)
        # self.displacements = torch.nn.Parameter(initial_guess.clone().detach())

    def forward(self, initial_guess):
        #Shape of initial guess is feature x X x Y
        guessReshape = initial_guess.reshape(initial_guess.shape[0], -1)
        #shape of reshape is going to be feature x XY
        prediction = self.l1(guessReshape)
        return prediction.reshape(initial_guess.shape) #reshape it back into the original shape


def displacement_loss(source_img, target_img, predicted_field):
    """
    Calculation of the loss function of the displacement field based on the
    intensity difference
    :param source_img: source image
    :param target_img: target image
    :param predicted_field: predicted displacement field
    :return: The loss related to intensity difference
    """
    predicted_image = displace_image(source_img, predicted_field, tensor=True)
    return torch.mean(torch.square(predicted_image - target_img))


def optical_template_displace_loss(optical_flow, template_flow, lambda_vel=0.5):
    """
    Calculation of the MSE loss function of the displacement field based on the
    difference between predicted field (optical) and the known field (template)
    :param optical_flow: predicted flow field
    :param template_flow: known flow field at intersections
    :param lambda_vel: regularizor value
    :return: The MSE loss associated with difference from predicted field to known field
    """
    squared_error = 0

    for x, y, dx, dy in template_flow:
        x_optical = optical_flow[x, y, 0]
        y_optical = optical_flow[x, y, 1]

        x_diff_squared = (x_optical - dx) ** 2
        y_diff_squared = (y_optical - dy) ** 2

        squared_error += (x_diff_squared + y_diff_squared)

    return lambda_vel * (squared_error / len(template_flow))


# # Function to compute total variation regularization
# def total_variation_regularization(displacement_field):
#     tv_loss = torch.sum(torch.abs(displacement_field[:, :, :, :-1] - displacement_field[:, :, :, 1:])) + \
#               torch.sum(torch.abs(displacement_field[:, :, :-1, :] - displacement_field[:, :, 1:, :]))
#     return tv_loss


# Step 3: Optimization loop
def optimize_displacement_field(model, source_img, target_img, observed_displacement,
                                optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Forward pass: compute predicted displacements
        initial_guess = torch.rand((2, 256,256))
        model = DisplacementFieldModel()
        predicted_displacement = model(initial_guess)

        # Compute the loss
        loss_intensity = displacement_loss(source_img, target_img, predicted_displacement)
        loss_displace = optical_template_displace_loss(predicted_displacement, observed_displacement)
        loss = loss_intensity + loss_displace
        print(loss, epoch)

        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


"""
Driver Code
"""
# Load original image and displacement field (example)
source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Target/synethetic_0.png'
template_path = 'Data/Template/frame_1_temp.png'
intersection = 'Data/Template/intersection.txt'
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)
template_image = cv2.imread(template_path)

of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()

source = TemplateMatching.TemplateMatcher(source_path, template_path, intersection)
target = TemplateMatching.TemplateMatcher(target_path, template_path, intersection)
source.match_template()
target.match_template()

observed = correspondence_displacement(source.matching_displacement(target))
predicted = of_object.get_flow()

# ----------------------- Optimization ---------------------------------------------------------------
# Initialize your displacement field model and optimizer
model = DisplacementFieldModel(predicted)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Optimize the displacement field
optimize_displacement_field(model, source_image, target_image,
                            observed, optimizer)

with torch.no_grad():
    displacement_field = model().cpu().detach().numpy()

visualize_displacement(source_image, displacement_field)

# matched_points_source = np.array([source.get_x_coord(), source.get_y_coord()])
# matched_points_target = np.array([target.get_x_coord(), target.get_y_coord()])

# Plotting
# plt.figure(figsize=(8, 6))
# plt.scatter(matched_points_source[0], matched_points_source[1], c='b', label='Source Points')
# plt.scatter(matched_points_target[0], matched_points_target[1], c='r', label='Target Points')

# Plot vectors
# for i in range(len(correspondence)):
#     plt.scatter(correspondence[i][0][0], correspondence[i][0][1],
#                 color='blue', marker='o')
#     plt.arrow(correspondence[i][0][0], correspondence[i][0][1],
#               correspondence[i][1][0] - correspondence[i][0][0],
#               correspondence[i][1][1] - correspondence[i][0][1],
#               head_width=2, head_length=4, fc='red', ec='red')
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()
# plt.show()
