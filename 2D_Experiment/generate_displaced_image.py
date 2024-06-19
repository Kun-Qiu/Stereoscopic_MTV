import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


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
    :return:                The transformed image
    """

    h, w = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_translate], [0, 1, y_translate]])
    translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

    # Displacement Field
    displacement_field = np.zeros((*image.shape, 2))
    displacement_field[..., 0] = x_translate
    displacement_field[..., 1] = y_translate

    return translated_image


def parabolic_transform(image, factor=0.0002):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    output_image = np.zeros_like(image)
    displacement_field = np.zeros((height, width, 2))

    for y in range(height):
        for x in range(width):
            # Distance of the pixel from the center
            dx, dy = x - center_x, y - center_y
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Compute the rotation angle proportional to the distance
            angle = factor * distance

            # Compute the perpendicular rotation
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            # Perpendicular direction to the radial vector (rotated by 90 degrees)
            perpendicular_dx = -dy
            perpendicular_dy = dx

            # Apply the perpendicular rotation
            new_x = int(center_x + cos_angle * perpendicular_dx - sin_angle * perpendicular_dy)
            new_y = int(center_y + sin_angle * perpendicular_dx + cos_angle * perpendicular_dy)

            displacement_field[y, x] = [new_x - x, new_y - y]

            # Only use valid coordinates within the image bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                output_image[y, x] = image[new_y, new_x]

    magnitude = np.linalg.norm(displacement_field, axis=-1)

    # Normalize displacement field to unit vectors, avoid division by zero
    displacement_field_unit = np.zeros_like(displacement_field)
    nonzero_magnitude = magnitude > 0
    displacement_field_unit[nonzero_magnitude] = (displacement_field[nonzero_magnitude] /
                                                  np.expand_dims(magnitude[nonzero_magnitude], axis=-1))

    # Plotting the quiver plot
    plt.figure(figsize=(8, 6))

    # Create a grid of points to plot
    X, Y = np.meshgrid(np.arange(0, width, 10), np.arange(0, height, 10))
    U = displacement_field_unit[::10, ::10, 0]
    V = displacement_field_unit[::10, ::10, 1]
    C = magnitude[::10, ::10]  # Magnitude for color

    plt.quiver(X, Y, U, V, C, cmap='viridis', scale=20, scale_units='inches')
    plt.colorbar(label='Magnitude')
    plt.title('Unit Vector Displacement Field')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('equal')
    plt.show()

    return output_image


# Define the parameters
fwhm = 4  # Full width at half maximum for the Gaussian lines
spacing = 25  # Spacing between the lines
angle = 60  # Angle in degrees for the intersecting lines
image_size = (512, 512)  # Size of the image

# Create the grid image
image = create_grid(image_size, fwhm, spacing, angle, snr=8)
translated_image = translate_image(image, x_translate=5, y_translate=5)
rotated_image = parabolic_transform(image)

# Display the image
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Gaussian Grid Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(translated_image, cmap='gray')
plt.title('Translated Grid Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(rotated_image, cmap='gray')
plt.title('Parabolic Transdformation Grid Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the image if needed
cv2.imwrite('gaussian_grid.png', image)
cv2.imwrite('gaussian_grid_trans.png', image)
cv2.imwrite('Poiseuille flow.png', rotated_image)
