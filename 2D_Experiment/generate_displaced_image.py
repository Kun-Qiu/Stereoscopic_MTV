import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt


def add_noise(image_array, snr):
    """
    Add noise to the image based on the specified SNR.

    image_array:    Input image array
    snr:            The desired signal-to-noise ratio
    """

    noise = np.random.random(image_array.size).reshape(*image_array.shape)
    current_snr = np.max(image_array) / (4 * np.std(noise))
    noise_img = image_array + (noise * (current_snr / snr))
    return noise_img


def gaussian_kernel(size, fwhm):
    """
    Generates a 1D Gaussian kernel for simulating the laser lines

    :param size:    Width of the laser line
    :param fwhm:    Full-width half-max width of the laser line
    :return:        Normalized gaussian mask
    """

    x = np.linspace(-size // 2, size // 2, size)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-x ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


def draw_gaussian_line(image, start_point, end_point, fwhm, intensity=3.0):
    """
    Draws a Gaussian line on the image with specified intensity.

    :param image:           The image where laser lines are drawn
    :param start_point:     The start point of the laser lines
    :param end_point:       The end point of the laser lines
    :param fwhm:            Full-width half-max width of the laser line
    :param intensity:       Intensity of the laser lines
    :return:                None
    """

    line_img = np.zeros_like(image)
    cv2.line(line_img, start_point, end_point, 255, 1)
    size = int(4 * fwhm)
    kernel = gaussian_kernel(size, fwhm)[:, None]
    gaussian_line_img = cv2.filter2D(line_img, -1, kernel)
    gaussian_line_img = cv2.filter2D(gaussian_line_img, -1, kernel.T)
    np.clip(gaussian_line_img * intensity, 0, 255, out=gaussian_line_img)
    np.add(image, gaussian_line_img, out=image, casting="unsafe")


def create_grid(image_size, fwhm, spacing, angle, line_intensity=1.0, snr=20):
    """
    Creates the grid image with Gaussian lines.

    :param image_size:          The size of the image (Length X Width)
    :param fwhm:                Full-width half-max width of the laser line
    :param spacing:             Spacing between the parallel laser lines
    :param angle:               Angle between two intersecting laser lines
    :param line_intensity:      Intensity of the laser lines
    :param snr:                 Signal to Noise ratio
    :return:                    An image with the simulated laser lines
    """

    image = np.zeros(image_size, dtype=np.float32)
    h, w = image_size
    radians = np.deg2rad(angle)

    # Draw lines at +60 degrees
    for x in range(-w, 2 * w, spacing):
        y1 = 0
        x1 = x
        y2 = h
        x2 = x + int(h / np.tan(radians))
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

    # Draw lines at -60 degrees
    for x in range(-w, 2 * w, spacing):
        y1 = 0
        x1 = x
        y2 = h
        x2 = x - int(h / np.tan(radians))
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

    # Normalize the image to [0, 255] and convert to uint8
    image = add_noise(np.clip(image, 0, 255).astype(np.uint8), snr)
    return image


def translate_image(image, x_translate=0, y_translate=0):
    """
    Translate the image by x_translate and y_translate amount

    :param image:           The input image of which the transformation is performed
    :param x_translate:     The number of pixel to translate in the x direction
    :param y_translate:     The number of pixel to translate in the y direction
    :return:                The translated image along with the displacement field
    """

    h, w = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

    # Displacement Field
    displacement_field = np.zeros((*image.shape, 2))
    displacement_field[..., 0] = x_translate
    displacement_field[..., 1] = y_translate

    return translated_image, displacement_field


def rotate_points(points, center, angle):
    """
    Rotate a set of points around a center by a certain angle

    :param points:      The points that the transformation is applied to
    :param center:      The center of the vortex
    :param angle:       Angle of rotation
    :return:            Set of points after the rotation
    """
    # Create a rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Translate points to origin
    translated_points = points - center

    # Apply rotation
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back
    result_points = rotated_points + center

    return result_points


def simulate_rotational_flow(image, max_rotation):
    """
    Transform an image by a maximum angle

    :param image:           The input image
    :param max_rotation:    The maximum rotation in radian
    :return:                An rotated image of the laser along with the displacement field
    """

    height, width = image.shape[:2]
    center = np.array([width // 2, height // 2])
    displacement_field = np.zeros((height, width, 2))

    y_coords, x_coords = np.indices((height, width))
    points = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)
    source_points = points.reshape((height, width, 2))

    # Compute distances to center
    distances_to_center = np.linalg.norm(points - center, axis=-1)
    max_distance = np.linalg.norm(center)
    relative_distances = distances_to_center / max_distance
    angles = max_rotation * relative_distances

    # Apply rotation
    rotated_points = np.zeros_like(points, dtype=float)
    for i, angle in enumerate(angles):
        rotated_points[i] = rotate_points(points[i], center, angle)

    # Map points back to image grid with nearest neighbor interpolation
    rotated_points = rotated_points.astype(int)
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], 0, width - 1)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], 0, height - 1)

    x_coords_rotated = rotated_points[:, 0].reshape(height, width)
    y_coords_rotated = rotated_points[:, 1].reshape(height, width)
    output_image = image[y_coords_rotated, x_coords_rotated]

    # Calculate displacement field
    displacement_field[:, :, 0] = x_coords_rotated - source_points[:, :, 0]
    displacement_field[:, :, 1] = y_coords_rotated - source_points[:, :, 1]

    return output_image, displacement_field


# ------------------------------------- Main Loop -------------------------------------------------------------

# Define the parameters
fwhm = 4  # Full width at half maximum for the Gaussian lines
spacing = 25  # Spacing between the lines
angle = 120  # Angle in degrees for the intersecting lines
image_size = (256, 256)  # Size of the image

snr_values = [1, 2, 4, 8, 16]
num_images_per_snr = 10
output_dir = "Experiment"
os.makedirs(output_dir, exist_ok=True)

for snr in snr_values:
    snr_folder = os.path.join(output_dir, f"SNR_{snr}")
    os.makedirs(snr_folder, exist_ok=True)

    for i in range(num_images_per_snr):
        grid_folder = os.path.join(snr_folder, f"Set_{i}")
        os.makedirs(grid_folder, exist_ok=True)  # Ensure the folder is created

        image = create_grid(image_size, fwhm, spacing, angle, line_intensity=2, snr=snr)
        translated_image, translated_field = translate_image(
            image,
            x_translate=random.randint(0, 10),
            y_translate=random.randint(0, 10)
        )
        rotated_image, rotated_field = simulate_rotational_flow(
            image,
            max_rotation=random.uniform(0, 0.2)
        )

        # Save images
        cv2.imwrite(os.path.join(grid_folder, f"Gaussian_Grid_Image_Set_{i}.png"), image)
        cv2.imwrite(os.path.join(grid_folder, f"Translational_Flow_Image_Set_{i}.png"), translated_image)
        cv2.imwrite(os.path.join(grid_folder, f"Rotational_Flow_Image_Set_{i}.png"), rotated_image)

        # Save displacement fields
        np.save(os.path.join(grid_folder, f"Translated_Field_Set_{i}.npy"), translated_field)
        np.save(os.path.join(grid_folder, f"Rotated_Field_Set_{i}.npy"), rotated_field)
