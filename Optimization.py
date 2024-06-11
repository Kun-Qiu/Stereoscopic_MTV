import cv2
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt
from Utility import Visualization as vis
import torch
import torch.nn.functional as F
from Utility.generateDisplacedImage import translateImage


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

    for vector in template_flow:
        x_template = vector[0][0]
        y_template = vector[0][1]
        x_displace_template = vector[1]
        y_displace_template = vector[2]

        x_optical = optical_flow[x_template, y_template, 0]
        y_optical = optical_flow[x_template, y_template, 1]

        x_diff_squared = (x_optical - x_displace_template) ** 2
        y_diff_squared = (y_optical - y_displace_template) ** 2

        squared_error += (x_diff_squared + y_diff_squared)

    return lambda_vel * (squared_error / len(template_flow))


def optimize_displacement_field(model, source_img, target_img, observed_displacement,
                                optimizer, lambda_int, lambda_vel, num_epochs):
    """
    The main training cycle for finding the solution that minimize the loss
    :param lambda_int: Regularizor for intensity
    :param lambda_vel: Regularizor for velocity
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
                                              lambda_intensity=lambda_int)

        loss_displace = known_displace_constraint(predicted_displacement,
                                                  observed_displacement,
                                                  lambda_vel=lambda_vel)

        loss_smooth = smoothness_constraint(u_displacement,
                                            v_displacement)

        loss = loss_smooth + loss_intensity + loss_displace

        if abs(loss - prevLoss) < 0.01:
            converged = True
        else:
            prevLoss = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch = epoch + 1
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return predicted_displacement


# --------------------------------------------------------------------------------------

"""
Driver Code
"""

# Load original image and displacement field (example)
source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Synthetic Target/synthetic_0.png'

# Optical Flow
of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()
of_object.visualize_flow()

# Template Matching
template_object = TemplateMatching.TemplateMatcher(source_path, target_path)
template_object.match_template_driver()

predicted = of_object.get_flow()
observed = template_object.get_displacement()

# ----------------------- Optimization ---------------------------------------------------------------
# Initialize displacement field model and optimizer
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

model = DisplacementFieldModel(predicted)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
optimized_displacement = optimize_displacement_field(model, source_image_tensor,
                                                     target_image_tensor,
                                                     observed, optimizer,
                                                     200, 1800, 10000)

vis.visualize_displacement(source_image, "Optimized Displacement", optimized_displacement)
vis.visualize_displacement(source_image, "Initial Displacement", predicted)
vis.visualize_displacement_difference(optimized_displacement, predicted, source_image)
plt.show()
