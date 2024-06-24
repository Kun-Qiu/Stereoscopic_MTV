import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_displacement(image, name, field):
    """
    Visualize the displacement vectors on top of the original image
    :param image: Original image
    :param name: Name for the plot
    :param field: Displacement field
    """
    field = torch.tensor(field)
    magnitudes = torch.sqrt(torch.sum(field ** 2, dim=2))

    magnitudes_numpy = magnitudes.detach().numpy()
    field_numpy = field.detach().numpy()

    length, height, color = image.shape

    plt.figure(figsize=(8, 6))
    # plt.imshow(image, cmap='gray')

    step = 10
    plt.quiver(range(0, length, 10), range(0, height, 10),
               field_numpy[::step, ::step, 0], field_numpy[::step, ::step, 1],
               magnitudes_numpy[::step, ::step],
               angles='xy', scale_units='xy', scale=1, cmap='viridis')
    plt.colorbar()
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')


def visualize_displacement_difference(field1, field2, image):
    """
    Visualize the difference between two displacement fields overlayed on the original image
    :param field1: First displacement field
    :param field2: Second displacement field
    :param image: Image data
    """
    # Compute the difference field
    if torch.is_tensor(field1):
        field1 = field1.detach().numpy()
    if torch.is_tensor(field2):
        field2 = field2.detach().numpy()
    diff_field = field2 - field1

    # Compute magnitudes
    magnitudes = np.sqrt(np.sum(diff_field ** 2, axis=2))

    # Plot the magnitude differences overlayed on the original image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.imshow(magnitudes, cmap='RdBu', alpha=0.5, interpolation='nearest', origin='lower')
    plt.colorbar(label='Magnitude of Difference')
    plt.title('Displacement Field Difference Overlayed on Image')
    plt.axis('off')
    plt.gca().invert_yaxis()
