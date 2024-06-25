import torch
import torch.nn.functional as F
from Utility.Optimization_Utility import translateImage


class DisplacementFieldModel(torch.nn.Module):
    def __init__(self, initial_guess):
        """
        Default constructor for the optimization model

        :param initial_guess:   The initial guess for the velocity field
        """
        super(DisplacementFieldModel, self).__init__()
        self.u = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 0],
                                                 dtype=torch.float32,
                                                 requires_grad=True))
        self.v = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 1],
                                                 dtype=torch.float32,
                                                 requires_grad=True))

    def forward(self):
        return torch.stack((self.u, self.v), dim=-1)


def smoothness_constraint(u, v, lambda_smooth=100):
    """
    Smoothness constraint for the field based on Horn Schunck's algorithm

    :param lambda_smooth:   Regularizor coefficient for smoothness of field
    :param u:               u component of displacement field
    :param v:               v component of displacement field
    :return:                Loss based on the gradient across x and y for u and v
    """
    du_x = u[:, :-1] - u[:, 1:]  # Difference on row elements
    du_y = u[:-1, :] - u[1:, :]  # Difference on column elements

    dv_x = v[:, :-1] - v[:, 1:]
    dv_y = v[:-1, :] - v[1:, :]

    # Pad the missing element for all four gradient to obtain n x n tensor
    du_x = F.pad(du_x, (0, 1), mode='constant', value=0)
    du_y = F.pad(du_y, (0, 0, 0, 1), mode='constant', value=0)
    dv_x = F.pad(dv_x, (0, 1), mode='constant', value=0)
    dv_y = F.pad(dv_y, (0, 0, 0, 1), mode='constant', value=0)

    smoothness_loss = torch.mean((du_x + du_y) ** 2 + (dv_x + dv_y) ** 2)
    return lambda_smooth * smoothness_loss


def intensity_constraint(source_img, target_img, predicted_field):
    """
    Calculation of the loss function of the displacement field based on the
    intensity difference

    :param source_img:          The source image (0)
    :param target_img:          The target image (Delta t)
    :param predicted_field:     Predicted displacement field
    :return:                    The mean square error related to intensity difference between
                                warped image and the target image
    """
    predicted_image = translateImage(source_img, predicted_field).squeeze().permute(1, 2, 0)

    # Ensure the predicted_image has the same shape as target_img
    if predicted_image.shape != target_img.shape:
        raise ValueError(f"Shape mismatch: predicted_image shape {predicted_image.shape} "
                         f"does not match target_img shape {target_img.shape}")

    predicted_img_norm = predicted_image
    target_img_norm = target_img
    return torch.mean(torch.square(target_img_norm - predicted_img_norm))


def intensity_gradient_constraint(source_img, target_img, predicted_field, lambda_intensity_grad=10.0):
    """
    Constraint based on the intensity gradient of the image. For a predicted image to be similar to the
    target image, the intensity gradient must be similar.

    :param source_img:              The source image (0)
    :param target_img:              The target image (Delta t)
    :param predicted_field:         Predicted displacement field
    :param lambda_intensity_grad:   Regularizor coefficient
    :return:                        The mean square error related to intensity difference between
                                    warped image and the target image
    """
    predicted_image = translateImage(source_img, predicted_field).squeeze().permute(1, 2, 0)

    # Ensure the predicted_image has the same shape as target_img
    if predicted_image.shape != target_img.shape:
        raise ValueError(f"Shape mismatch: predicted_image shape {predicted_image.shape} "
                         f"does not match target_img shape {target_img.shape}")

    I_predicted = torch.gradient(predicted_image, spacing=1.0)
    I_target = torch.gradient(target_img, spacing=1.0)

    div_predicted = I_predicted[0] + I_predicted[1]
    div_target = I_target[0] + I_target[1]

    loss_intensity_grad = torch.mean((torch.abs(div_predicted - div_target)) ** 2)
    return lambda_intensity_grad * loss_intensity_grad


def known_displace_constraint(optical_flow, template_flow, lambda_vel=10.0):
    """
    Calculation of the MSE loss function of the displacement field based on the
    difference between predicted field (optical) and the known field (template)

    :param optical_flow:        The predicted flow field
    :param template_flow:       The known flow field at intersections (Template Matching)
    :param lambda_vel:          Regularizotion coefficient
    :return:                    The MSE loss associated with difference from
                                predicted field to known field
    """
    squared_error = torch.tensor(0.0)

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

    return lambda_vel * (torch.abs(squared_error) / len(template_flow))


def optimize_displacement_field(model, source_img, target_img, observed_displacement,
                                optimizer, lambda_smooth=100, lambda_int_grad=5, lambda_vel=25,
                                num_epochs=10000):
    """
    The main training cycle for finding the solution that minimize the
    total loss function

    :param lambda_smooth:           Regularization coefficient for smoothness of the field
    :param lambda_int_grad:         Regularization coefficient for intensity gradient across images
    :param lambda_vel:              Regularization coefficient for velocity difference
    :param model:                   The model object in which to train
    :param source_img:              Initial image of the flow @ t=0
    :param target_img:              Displaced image @ t=dt
    :param observed_displacement:   Initial velocity field (Template Matching)
    :param optimizer:               Type of optimizer (Default=Adams)
    :param num_epochs:              The number of epoch
    :return:                        The optimized velocity field
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
                                              predicted_displacement)

        loss_intensity_gradient = intensity_gradient_constraint(source_img,
                                                                target_img,
                                                                predicted_displacement,
                                                                lambda_intensity_grad=lambda_int_grad)

        loss_displace = known_displace_constraint(predicted_displacement,
                                                  observed_displacement,
                                                  lambda_vel=lambda_vel)

        loss_smooth = smoothness_constraint(u_displacement, v_displacement, lambda_smooth)

        loss = loss_smooth + loss_intensity + loss_displace + loss_intensity_gradient

        if torch.abs(loss - prevLoss) < 0.01:
            converged = True
        else:
            prevLoss = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch = epoch + 1
        print(f"intensity: {loss_intensity}, intensity gradient: {loss_intensity_gradient}, "
              f"displace: {loss_displace}, smooth: {loss_smooth}")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return predicted_displacement
