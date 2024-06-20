import torch.nn.functional as F
import torch


def translateImage(image, translateField):
    """
    Transform the input image by a displacement field

    :param image:               The source image
    :param translateField:      The displacement field
    :return:                    Transformed image
    """
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
