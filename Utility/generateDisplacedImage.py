import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


def generatingDisplacement(img_size, dx, dy, mode):
    height, width, channels = img_size

    # 2D array to store displacement vectors
    displacement_field = np.zeros((height, width, 2), dtype=float)
    for y in range(height):
        for x in range(width):
            if mode == 'constant':
                displacement_field[x, y] = [dy, dx]
            elif mode == 'vortex':
                # Calculate vortex speed based on distance from the center
                distance_from_center = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
                vortex_speed = 1 / (1 + distance_from_center)  # Adjusted vortex formula
                displacement_field[x, y] = [dx * vortex_speed, dy * vortex_speed]

    return displacement_field


def display_displacement_field(displacement_field, subsampling_factor=10):
    # Extract displacement components
    dx_plot = displacement_field[:, :, 1]

    """
    Due to the coordinate system start from top left on an image,
    y displacement need to be inverted
    """
    dy_plot = -1 * displacement_field[:, :, 0]

    magnitude = np.sqrt(dx_plot ** 2 + dy_plot ** 2)

    # Create grid coordinates for quiver plot
    y_plot, x_plot = np.mgrid[0:dx_plot.shape[0], 0:dx_plot.shape[1]]

    # Subsample grid coordinates and displacement field
    y_plot_subsampled = y_plot[::subsampling_factor, ::subsampling_factor]
    x_plot_subsampled = x_plot[::subsampling_factor, ::subsampling_factor]
    magnitude_subsampled = magnitude[::subsampling_factor, ::subsampling_factor]
    dx_plot_subsampled = dx_plot[::subsampling_factor, ::subsampling_factor]
    dy_plot_subsampled = dy_plot[::subsampling_factor, ::subsampling_factor]

    # Plot magnitude with colorbar
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, extent=[0, dx_plot.shape[1], 0, dx_plot.shape[0]], cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Displacement Field Magnitude')

    # Plot unit vectors for direction
    dx_uni_vector = dx_plot_subsampled / magnitude_subsampled
    dy_uni_vector = dy_plot_subsampled / magnitude_subsampled
    plt.quiver(x_plot_subsampled, y_plot_subsampled, dx_uni_vector, dy_uni_vector,
               scale=50, color='white')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Could not be used with PyTorch
def displace_image(input_img, min_value, max_value, mode='constant'):
    image = cv2.imread(input_img)
    height, width, channels = image.shape

    # Generate displacement field
    displacement_field = generatingDisplacement(image.shape, min_value, max_value, mode)

    dx = displacement_field[:, :, 0]
    dy = displacement_field[:, :, 1]

    displaced_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            new_x = int(x + dx[x, y])
            new_y = int(y + dy[x, y])

            if 0 <= new_x < width and 0 <= new_y < height:
                displaced_image[new_x, new_y] = image[x, y]

    return displacement_field, displaced_image


# Used for PyTorch
def translateImage(image, translateField):
    """
    Displace the source image by the displacement field pixel by pixel
    :param image: source image
    :param translateField: displacement field
    :return: Displaced image
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


def main(input_img, target_path, min_value, max_value, mode):
    try:
        field, displaced_image = displace_image(input_img, min_value, max_value, mode)

        if displaced_image is not None:
            cv2.imwrite(target_path, displaced_image)
            print(f"Image generated and saved at: {target_path}")
            display_displacement_field(field)
        else:
            print("Error: Failed to displace image.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Displace an image.')
    parser.add_argument('input_img', type=str, help='Path to the input image')
    parser.add_argument('target_path', type=str, help='Path to save image')
    parser.add_argument('--dx', type=int, default=-3, help='Minimum displacement value')
    parser.add_argument('--dy', type=int, default=3, help='Maximum displacement value')
    parser.add_argument('--mode', type=str, default='constant', choices=['constant', 'vortex'],
                        help='Mode for displacement')
    args = parser.parse_args()
    main(args.input_img, args.target_path, args.dx, args.dy, args.mode)
