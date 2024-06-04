import cv2
import numpy as np
import torch
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Utility Functions
def translateImage(image, translateField):
    """
    #     Displace the source image by the displacement field pixel by pixel
    #     :param image: source image
    #     :param translateField: displacement field
    #     :return: Displaced image
    #   """
    length, width, channel = image.shape

    size = [length, width]
    x_r = torch.arange(size[0])
    y_r = torch.arange(size[1])
    x_grid, y_grid = torch.meshgrid(x_r, y_r)  # create the original grid

    field_x = translateField[:, :, 1]  # obtain the translation field for x and y independently
    field_y = translateField[:, :, 0]

    translate_x = (x_grid + field_x) / size[0] * 2 - 1  # Translate the original coordinate space
    translate_y = (y_grid - field_y) / size[1] * 2 - 1

    tFieldXY = torch.stack((translate_y, translate_x)).permute(1, 2, 0).unsqueeze(0)

    img = image.permute(2, 0, 1).unsqueeze(0)
    output = F.grid_sample(img, tFieldXY, padding_mode='zeros')
    return output


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

    def forward(self):
        return torch.stack((self.u, self.v), dim=-1)


def smoothness_constraint(u, v):
    """
    Smoothness constraint for the field based on Horn Schunck's algorithm
    :param u: u component of displacement field
    :param v: v component of displacement field
    :return: loss based on the gradient across x and y for u and v
    """
    # Calculate squared differences
    du_x = u[:, :-1] - u[:, 1:]  # Difference on row elements
    du_y = u[:-1, :] - u[1:, :]  # Difference on column elements

    dv_x = v[:, :-1] - v[:, 1:]  # Difference on row elements
    dv_y = v[:-1, :] - v[1:, :]  # Difference on column elements

    # Pad the missing element for all four gradient to obtain n x n tensor
    du_x = F.pad(du_x, (0, 1), mode='constant', value=0)
    du_y = F.pad(du_y, (0, 0, 0, 1), mode='constant', value=0)
    dv_x = F.pad(dv_x, (0, 1), mode='constant', value=0)
    dv_y = F.pad(dv_y, (0, 0, 0, 1), mode='constant', value=0)

    # Sum squared differences
    smoothness_loss = 0.25 * torch.sum(du_x ** 2 + du_y ** 2 + dv_x ** 2 + dv_y ** 2)
    return smoothness_loss


def intensity_constraint(source_img, target_img, predicted_field, lambda_intensity=10.0):
    """
    Calculation of the loss function of the displacement field based on the
    intensity difference
    :param source_img: source image
    :param target_img: target image
    :param predicted_field: predicted displacement field
    :param lambda_intensity: regularizor coefficient for the intensity loss
    :return: The loss related to intensity difference
    """
    predicted_image = translateImage(source_img, predicted_field).squeeze().permute(1, 2, 0)
    return lambda_intensity * torch.mean(torch.square(predicted_image - target_img))


def known_displace_constraint(optical_flow, template_flow, lambda_vel=10.0):
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


def optimize_displacement_field(model, source_img, target_img, observed_displacement,
                                optimizer, num_epochs=10000):
    """
    The main training cycle for finding the solution that minimize the loss
    :param model: model
    :param source_img: image @ t=0
    :param target_img: image @ t=dt
    :param observed_displacement: initial velocity field
    :param optimizer: optimizer
    :param num_epochs: number of epoch
    :return: optimized velocity field
    """

    predicted_displacement = None
    epoch = 0
    converged = False
    prevLoss = 0
    while epoch < num_epochs and not converged:
        predicted_displacement = model()

        u_displacement = predicted_displacement[:, :, 0]
        v_displacement = predicted_displacement[:, :, 1]

        predicted_displacement = predicted_displacement.view(256, 256, 2)
        loss_intensity = intensity_constraint(source_img,
                                              target_img,
                                              predicted_displacement,
                                              lambda_intensity=15.5)

        loss_displace = known_displace_constraint(predicted_displacement,
                                                  observed_displacement,
                                                  lambda_vel=275)

        loss_smooth = smoothness_constraint(u_displacement,
                                            v_displacement)

        loss = loss_smooth + loss_intensity + loss_displace

        if abs(loss - prevLoss) < 0.001:
            converged = True
        else:
            prevLoss = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch = epoch + 1
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return predicted_displacement


"""
Driver Code
"""
# Load original image and displacement field (example)
source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Target/frame_0_2us.png'
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

# # Plotting
# plt.figure(figsize=(8, 6))
# # plt.scatter(matched_points_source[0], matched_points_source[1], c='b', label='Source Points')
# # plt.scatter(matched_points_target[0], matched_points_target[1], c='r', label='Target Points')
#
# x_coord = []
# y_coord = []
# dx_val = []
# dy_val = []
# for x, y, dx, dy in observed:
#     x_coord.append(x)
#     y_coord.append(y)
#     dx_val.append([optimized_displacement[x, y][0].detach().numpy(), dx])
#     dy_val.append([optimized_displacement[x, y][1].detach().numpy(), dy])
#
# plt.scatter(x_coord, y_coord, c='b', label='Source Points')
#
# # Plot vectors
# for x, y, dx, dy in zip(x_coord, y_coord, dx_val, dy_val):
#     plt.arrow(x, y, dx[0], dy[0], head_width=2, head_length=4, fc='red', ec='red', label='Predicted')
#     plt.arrow(x, y, dx[1], dy[1], head_width=2, head_length=4, fc='blue', ec='blue', label='Known')
#
# # # Plot vectors
# # for i in range(len(correspondence)):
# #     plt.scatter(correspondence[i][0][0], correspondence[i][0][1],
# #                 color='blue', marker='o')
# #     plt.arrow(correspondence[i][0][0], correspondence[i][0][1],
# #               correspondence[i][1][0] - correspondence[i][0][0],
# #               correspondence[i][1][1] - correspondence[i][0][1],
# #               head_width=2, head_length=4, fc='red', ec='red')
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()
# plt.show()
