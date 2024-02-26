import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


def generatingDisplacement(img_size, min_value, max_value, mode):
    height, width, channels = img_size

    # Generate random displacement values within the specified range
    dx = np.random.randint(min_value, max_value + 1)
    dy = np.random.randint(min_value, max_value + 1)

    # 2D array to store displacement vectors
    displacement_field = np.zeros((height, width, 2), dtype=float)
    for y in range(height):
        for x in range(width):
            if mode == 'constant':
                displacement_field[y, x] = [dx, dy]
            elif mode == 'vortex':
                # Calculate vortex speed based on distance from the center
                distance_from_center = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
                vortex_speed = 1 / (1 + distance_from_center)  # Adjusted vortex formula
                displacement_field[y, x] = [dx * vortex_speed, dy * vortex_speed]
    return displacement_field


def display_displacement_field(displacement_field, subsampling_factor=10):
    # Extract displacement components
    dx_plot = displacement_field[:, :, 0]
    dy_plot = -1 * displacement_field[:, :, 1]  # Invert y-component

    # Calculate magnitude
    magnitude = np.sqrt(dx_plot**2 + dy_plot**2)

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
    dx_uni_vector = dx_plot_subsampled/magnitude_subsampled
    dy_uni_vector = dy_plot_subsampled/magnitude_subsampled
    plt.quiver(x_plot_subsampled, y_plot_subsampled, dx_uni_vector, dy_uni_vector ,
               scale=50, color='white')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


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
            new_x = int(x + dx[y, x])
            new_y = int(y + dy[y, x])

            if 0 <= new_x < width and 0 <= new_y < height:
                displaced_image[new_y, new_x] = image[y, x]

    return displacement_field, displaced_image


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
    parser.add_argument('--min_val', type=int, default=-3, help='Minimum displacement value')
    parser.add_argument('--max_val', type=int, default=3, help='Maximum displacement value')
    parser.add_argument('--mode', type=str, default='constant', choices=['constant', 'vortex'],
                        help='Mode for displacement')
    args = parser.parse_args()
    main(args.input_img, args.target_path, args.min_val, args.max_val, args.mode)