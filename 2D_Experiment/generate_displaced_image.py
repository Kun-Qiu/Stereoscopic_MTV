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
    for x in range(-w, w, spacing):
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
    return translated_image


def rotational_displacement_transform(image, max_translation=20, max_rotation=0.4):
    """
    Apply a rotational displacement to the input image.

    :param image:               The input image of which the transformation is performed
    :param max_translation:     Number of pixels for translational transformation
    :param max_rotation:        Number of degrees for rotational transformation
    :return:                    Simulated image representing a Poiseuille flow
    """
    image_array = np.array(image)

    def rotational_displacement(x, y, translation, rotation):
        # Center the coordinates
        x_center = x.max() / 2
        y_center = y.max() / 2
        x_centered = x - x_center
        y_centered = y - y_center

        # Calculate the radius and angle for each point
        radius = np.sqrt(x_centered ** 2 + y_centered ** 2)
        angle = np.arctan2(y_centered, x_centered)

        # Apply the rotational displacement
        angle_new = angle + rotation * (radius / radius.max())
        radius_new = radius + translation * (radius / radius.max())

        # Convert back to Cartesian coordinates
        x_new = radius_new * np.cos(angle_new) + x_center
        y_new = radius_new * np.sin(angle_new) + y_center

        return x_new, y_new

    # Get the original coordinates
    height, width = image_array.shape
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Flatten the coordinates
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Apply the rotational displacement
    x_new, y_new = rotational_displacement(x_flat, y_flat, max_translation, max_rotation)

    # Interpolate the displaced image
    displaced_image = griddata(
        (x_flat, y_flat),
        image_array.flatten(),
        (x_new, y_new),
        method='linear',
        fill_value=0
    ).reshape((height, width))

    return displaced_image


# Define the parameters
fwhm = 4  # Full width at half maximum for the Gaussian lines
spacing = 25  # Spacing between the lines
angle = 60  # Angle in degrees for the intersecting lines
image_size = (512, 512)  # Size of the image

# Create the grid image
image = create_grid(image_size, fwhm, spacing, angle, snr=1)
translated_image = translate_image(image, x_translate=20, y_translate=20)
rotated_image = rotational_displacement_transform(image)

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
