import torch
import torch.nn.functional as F
from Utility.Optimization_Utility import translateImage
import torchvision.transforms.functional as func


def rgb_to_grayscale(rgb_image):
    """
    Convert an RGB image tensor to grayscale using torchvision.

    :param rgb_image: A 3-channel (RGB) image tensor of shape (H, W, 3) or (N, H, W, 3)
    :return: A grayscale image tensor of shape (H, W) or (N, H, W)
    """
    if rgb_image.ndimension() == 3:
        # If input is a single image of shape (H, W, 3)
        grayscale_image = func.rgb_to_grayscale(rgb_image.permute(2, 0, 1), num_output_channels=1).squeeze(0)
        return grayscale_image
    elif rgb_image.ndimension() == 4:
        # If input is a batch of images of shape (N, H, W, 3)
        grayscale_batch = torch.stack(
            [func.rgb_to_grayscale(img.permute(2, 0, 1), num_output_channels=1).squeeze(0) for img in rgb_image])
        return grayscale_batch
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions (H, W, 3) or (N, H, W, 3)")


class DisplacementFieldModel(torch.nn.Module):
    def __init__(self, initial_guess, hidden_size=128):
        """
        Default constructor for the optimization model

        :param initial_guess:   The initial guess for the velocity field
        :param hidden_size:     Size of the hidden layer
        """
        super(DisplacementFieldModel, self).__init__()
        self.u = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 0],
                                                 dtype=torch.float32,
                                                 requires_grad=True))
        self.v = torch.nn.Parameter(torch.tensor(initial_guess[:, :, 1],
                                                 dtype=torch.float32,
                                                 requires_grad=True))

        # Adding a fully connected layer
        self.fc1 = torch.nn.Linear(256 * 256 * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 256 * 256 * 2)

    def forward(self):
        uv = torch.stack((self.u, self.v), dim=-1).view(-1)
        uv = F.relu(self.fc1(uv))
        uv = self.fc2(uv)
        uv = uv.view(256, 256, 2)
        return uv


def smoothness_constraint(u, v, lambda_smooth=100):
    """
    Smoothness constraint for the field

    :param u:               u component of displacement field
    :param v:               v component of displacement field
    :param lambda_smooth:   Regularizor coefficient for smoothness of field
    :return:                Loss based on the gradient across x and y for u and v
    """
    squared_ep = torch.tensor(0.001)

    du_x = u[:, :-1] - u[:, 1:]  # Difference on row elements
    du_y = u[:-1, :] - u[1:, :]  # Difference on column elements

    dv_x = v[:, :-1] - v[:, 1:]
    dv_y = v[:-1, :] - v[1:, :]

    # Pad the missing element for all four gradient to obtain n x n tensor
    du_x = F.pad(du_x, (0, 1), mode='constant', value=0)
    du_y = F.pad(du_y, (0, 0, 0, 1), mode='constant', value=0)
    dv_x = F.pad(dv_x, (0, 1), mode='constant', value=0)
    dv_y = F.pad(dv_y, (0, 0, 0, 1), mode='constant', value=0)

    smoothness_loss = torch.sum(torch.sqrt((du_x ** 2 + du_y ** 2 + dv_x ** 2 + dv_y ** 2) + squared_ep))
    return lambda_smooth * smoothness_loss


def energy_data(source_img, target_img, predicted_field, lambda_grad=10):
    """
    Calculation of the energy based on the global deviation of the grayscale intensity of
    the warped image from the ground truth. In addition, the gradient constancy between
    warped image and the target image is calculated. L1 norm is used to minimize influence
    of outliers in the field approximation

    :param source_img:          The source image (0)
    :param target_img:          The target image (Delta t)
    :param predicted_field:     Predicted displacement field
    :param lambda_grad:         Regularizor coefficient for the intensity gradient
    :return:                    The mean square error related to intensity difference between
                                warped image and the target image
    """
    ep_squared = torch.square(torch.tensor(0.001))
    predicted_img = translateImage(source_img, predicted_field).squeeze().permute(1, 2, 0)

    predicted_img_gray = rgb_to_grayscale(predicted_img)
    target_img_gray = rgb_to_grayscale(target_img)

    # Gradient Constancy Assumption
    grad_predicted = torch.gradient(predicted_img_gray, spacing=1.0)
    grad_target = torch.gradient(target_img_gray, spacing=1.0)

    # Ensure the predicted_image has the same shape as target_img
    if predicted_img.shape != target_img.shape:
        raise ValueError(f"Shape mismatch: predicted_image shape {predicted_img.shape} "
                         f"does not match target_img shape {target_img.shape}")

    grad_x = grad_predicted[0]
    grad_y = grad_predicted[1]
    grad_z = predicted_img - target_img
    grad_xz = torch.abs(grad_x - grad_target[0])
    grad_yz = torch.abs(grad_y - grad_target[1])

    intensity_loss = torch.sum(torch.sqrt(torch.square(torch.abs(grad_z) + ep_squared)))
    grad_loss = lambda_grad * torch.sum(torch.sqrt(torch.square(grad_xz) + torch.square(grad_yz) + ep_squared))
    data_loss = intensity_loss + grad_loss
    return data_loss


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
    ep_squared = torch.square(torch.tensor(0.001))

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

    return lambda_vel * torch.sqrt(squared_error + ep_squared)


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    while epoch < num_epochs and not converged:
        predicted_displacement = model()

        u_displacement = predicted_displacement[:, :, 0]
        v_displacement = predicted_displacement[:, :, 1]

        predicted_displacement = predicted_displacement.view(256, 256, 2)
        loss_data = energy_data(source_img,
                                target_img,
                                predicted_displacement,
                                lambda_int_grad)

        loss_displace = known_displace_constraint(predicted_displacement,
                                                  observed_displacement,
                                                  lambda_vel=lambda_vel)

        loss_smooth = smoothness_constraint(u_displacement, v_displacement,
                                            lambda_smooth)

        loss = loss_smooth + loss_data + loss_displace

        if torch.abs(loss - prevLoss) < 0.01:
            converged = True
        else:
            prevLoss = loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        epoch = epoch + 1
        print(f"data: {loss_data}, displace: {loss_displace}, smooth: {loss_smooth}")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    print(epoch)
    return predicted_displacement
