import cv2
import numpy as np


def denoised_image(input_image, threshold=8):
    # Perform Fourier Transform
    f_transform = np.fft.fft2(input_image)
    f_shift = np.fft.fftshift(f_transform)

    # Calculate magnitude spectrum
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)

    # Apply a threshold to filter out high-frequency noise
    f_shift[magnitude_spectrum < threshold] = 0

    # Inverse Fourier Transform
    inverse_transform = np.fft.ifftshift(f_shift)
    denoised_image = np.fft.ifft2(inverse_transform).real

    # Convert back to uint8 for image display
    denoised_image = np.uint8(denoised_image)

    return denoised_image
