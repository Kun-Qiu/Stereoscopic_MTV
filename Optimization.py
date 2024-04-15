import cv2
import numpy as np
import torch
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Utility Functions
def displace_image(image, field):
    """
    Displace the source image by the displacement field pixel by pixel
    :param image: source image
    :param field: displacement field
    :return: Displaced image
    """
    dx = field[:, :, 0]
    dy = field[:, :, 1]

    # print(source_img_clone.shape)
    # print(dx.shape)
    # for x in range(width):
    #     for y in range(height):
    #         displacement_x, displacement_y = field[x, y]
    #
    #         new_x = int(x + displacement_x)
    #         new_y = int(y + displacement_y)
    #         if 0 <= new_x < width and 0 <= new_y < height:
    #             displaced_image[x, y] = image[new_x, new_y]

    # Create grid of coordinates
    h, w, channel = image.size()

    # Create grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), dim=2).float()

    # Add displacement to grid
    shifted_grid = grid + torch.stack((dx, dy)).permute(1, 2, 0)

    # Apply bilinear interpolation
    shifted_image = F.grid_sample(image.squeeze(0), shifted_grid, mode='bilinear', padding_mode='zeros')
    print(shifted_image.shape)

    return shifted_image.squeeze(0)


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


def visualize_displacement(image, name, field):
    """
    Visualize the displacement vectors on top of the original image
    :param image: Original image
    :param name: Name for the plot
    :param field: Displacement field
    """
    field = torch.tensor(field)
    magnitudes = torch.sqrt(torch.sum(field ** 2, dim=2))

    magnitudes_numpy = magnitudes.detach().numpy()
    field_numpy = field.detach().numpy()

    length, height, color = image.shape

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')

    step = 10
    plt.quiver(range(0, length, 10), range(0, height, 10),
               field_numpy[::step, ::step, 0], field_numpy[::step, ::step, 1],
               magnitudes_numpy[::step, ::step],
               angles='xy', scale_units='xy', scale=1, cmap='viridis')
    plt.colorbar()
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')


def visualize_displacement_difference(field1, field2, image):
    """
    Visualize the difference between two displacement fields overlayed on the original image
    :param field1: First displacement field
    :param field2: Second displacement field
    :param image: Image data
    """
    # Compute the difference field
    if torch.is_tensor(field1):
        field1 = field1.detach().numpy()
    if torch.is_tensor(field2):
        field2 = field2.detach().numpy()
    diff_field = field2 - field1

    # Compute magnitudes
    magnitudes = np.sqrt(np.sum(diff_field ** 2, axis=2))

    # Plot the magnitude differences overlayed on the original image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(magnitudes, cmap='RdBu', alpha=0.5, interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of Difference')
    plt.title('Displacement Field Difference Overlayed on Image')
    plt.axis('off')
    plt.gca().invert_yaxis()


# __________________________________________________________________________________________

class DisplacementFieldModel(torch.nn.Module):
    def __init__(self, initial_guess):
        super(DisplacementFieldModel, self).__init__()

        self.u = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 0],
                                                 dtype=torch.float32,
                                                 requires_grad=True))
        self.v = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 1],
                                                 dtype=torch.float32,
                                                 requires_grad=True))
        # self.u = torch.nn.Parameter(torch.zeros(256, 256, 1), requires_grad=True)
        # self.v = torch.nn.Parameter(torch.zeros(256, 256, 1), requires_grad=True)

    def forward(self):
        return torch.stack((self.u, self.v), dim=-1)


def displacement_loss(source_img, target_img, predicted_field):
    """
    Calculation of the loss function of the displacement field based on the
    intensity difference
    :param source_img: source image
    :param target_img: target image
    :param predicted_field: predicted displacement field
    :return: The loss related to intensity difference
    """
    predicted_image = displace_image(source_img, predicted_field)
    result_np = predicted_image.detach().numpy()

    # Display the image
    plt.imshow(result_np)
    plt.axis('off')  # Turn off axis
    plt.show()

    return torch.mean(torch.square(predicted_image - target_img))


def optical_template_displace_loss(optical_flow, template_flow, lambda_vel=50):
    """
    Calculation of the MSE loss function of the displacement field based on the
    difference between predicted field (optical) and the known field (template)
    :param optical_flow: predicted flow field
    :param template_flow: known flow field at intersections
    :param lambda_vel: regularizor value
    :return: The MSE loss associated with difference from predicted field to known field
    """
    squared_error = torch.tensor(0, dtype=torch.float32)

    for x, y, dx, dy in template_flow:
        x_optical = optical_flow[x, y, 0]
        y_optical = optical_flow[x, y, 1]

        x_diff_squared = (x_optical - dx) ** 2
        y_diff_squared = (y_optical - dy) ** 2

        squared_error += (x_diff_squared + y_diff_squared)

    return lambda_vel * (squared_error / len(template_flow))


def gradient_loss(field, lambda_grad=1):
    # Compute gradients of u component with respect to x and y axes

    du_dx = torch.tensor(np.gradient(field.detach().numpy()[:, :, 0], axis=0))
    du_dy = torch.tensor(np.gradient(field.detach().numpy()[:, :, 0], axis=1))
    dv_dx = torch.tensor(np.gradient(field.detach().numpy()[:, :, 1], axis=0))
    dv_dy = torch.tensor(np.gradient(field.detach().numpy()[:, :, 1], axis=1))

    grad_norm_squared = torch.sum(du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2)
    return lambda_grad * grad_norm_squared


# # Function to compute total variation regularization
# def total_variation_regularization(displacement_field):
#     tv_loss = torch.sum(torch.abs(displacement_field[:, :, :, :-1] - displacement_field[:, :, :, 1:])) + \
#               torch.sum(torch.abs(displacement_field[:, :, :-1, :] - displacement_field[:, :, 1:, :]))
#     return tv_loss


def optimize_displacement_field(model, source_img, target_img, observed_displacement,
                                optimizer, num_epochs=10):
    predicted_displacement = None
    for epoch in range(num_epochs):
        predicted_displacement = model()

        predicted_displacement = predicted_displacement.view(256, 256, 2)
        loss_intensity = displacement_loss(source_img,
                                           target_img,
                                           predicted_displacement)

        # loss_displace = optical_template_displace_loss(predicted_displacement,
        #                                                observed_displacement,
        #                                                lambda_vel=50)

        # loss_grad = gradient_loss(predicted_displacement, lambda_grad=1)

        # print(f'inten: {loss_intensity}, vel: {loss_displace}, grad: {loss_grad}')
        # loss = loss_displace + loss_intensity + loss_grad
        # print(f'inten: {loss_intensity}, grad: {loss_grad}')
        loss = loss_intensity

        loss.backward()
        optimizer.step()
        # print('gradients =', [x.grad.data for x in model.parameters()])
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'Parameter: {name}, Size: {len(param.grad.nonzero())}, Gradient: {param.grad.nonzero()}')
                # print('weights after backpropagation = ', list(model.parameters()))
        optimizer.zero_grad()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return predicted_displacement


"""
Driver Code
"""
# Load original image and displacement field (example)
source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Target/synethetic_0.png'
template_path = 'Data/Template/frame_0_temp.png'
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
# source.visualizeMatchAfterNonMaxSuppression()
# target.visualizeMatchAfterNonMaxSuppression()

correspondence = source.matching_displacement(target)
observed = correspondence_displacement(correspondence)
predicted = of_object.get_flow()
# of_object.visualize_flow()

# ----------------------- Optimization ---------------------------------------------------------------

# Initialize displacement field model and optimizer
model = DisplacementFieldModel(predicted)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
optimized_displacement = optimize_displacement_field(model, source_image_tensor,
                                                     target_image_tensor,
                                                     observed, optimizer)

visualize_displacement(source_image, "Optimized Displacement", optimized_displacement)
visualize_displacement(source_image, "Initial Displacement", predicted)
visualize_displacement_difference(optimized_displacement, predicted, source_image)
plt.show()

# matched_points_source = np.array([source.get_x_coord(), source.get_y_coord()])
# matched_points_target = np.array([target.get_x_coord(), target.get_y_coord()])

# Plotting
plt.figure(figsize=(8, 6))
# plt.scatter(matched_points_source[0], matched_points_source[1], c='b', label='Source Points')
# plt.scatter(matched_points_target[0], matched_points_target[1], c='r', label='Target Points')

x_coord = []
y_coord = []
dx_val = []
dy_val = []
for x, y, dx, dy in observed:
    x_coord.append(x)
    y_coord.append(y)
    dx_val.append([optimized_displacement[x, y][0].detach().numpy(), dx])
    dy_val.append([optimized_displacement[x, y][1].detach().numpy(), dy])

plt.scatter(x_coord, y_coord, c='b', label='Source Points')

# Plot vectors
for x, y, dx, dy in zip(x_coord, y_coord, dx_val, dy_val):
    plt.arrow(x, y, dx[0], dy[0], head_width=2, head_length=4, fc='red', ec='red', label='Predicted')
    plt.arrow(x, y, dx[1], dy[1], head_width=2, head_length=4, fc='blue', ec='blue', label='Known')

# # Plot vectors
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
plt.show()
