import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_displacement(image, name, field, step, show_img=False, save_path=None):
    """
    Visualize the displacement vectors on top of the original image
    :param image:       Original image
    :param name:        Name for the plot
    :param field:       The displacement field
    :param step:        Step size for down-sampling the vector field
    :param show_img:    Whether to show the source image
    :param save_path:   Path in which the figures will be saved
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape  # Assuming a grayscale image

    field = torch.tensor(field)
    magnitudes = torch.sqrt(torch.sum(field ** 2, dim=2))

    magnitudes_numpy = magnitudes.detach().numpy()
    field_numpy = field.detach().numpy()

    length, height, color = image.shape

    plt.figure(figsize=(8, 6))
    if show_img:
        plt.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]], aspect='auto')

    if length == width:
        plt.quiver(range(0, length, step), range(0, height, step),
                   field_numpy[::step, ::step, 0], field_numpy[::step, ::step, 1],
                   magnitudes_numpy[::step, ::step],
                   angles='xy', scale_units='xy', scale=1, cmap='viridis')
    else:
        step_x = 9
        step_y = 16
        plt.quiver(range(0, height, step_y), range(0, length, step_x),
                   field_numpy[::step_x, ::step_y, 0], field_numpy[::step_x, ::step_y, 1],
                   magnitudes_numpy[::step_x, ::step_y],
                   angles='xy', scale_units='xy', scale=1, cmap='viridis')
    plt.colorbar()
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")


def visualize_displacement_difference(field1, field2, image, save_path=None):
    """
    Visualize the difference between two displacement fields overlay onto the original image
    :param field1:      First displacement field
    :param field2:      Second displacement field
    :param save_path:   Path in which the figures will be saved
    :param image:       Image data
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

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")


def plot_interpolation(XY, dXYZ, unit_label, contour=False, plot=True):
    """
    Plot the given dXYZ array whether it is 1D, 2D, 3D with the associated
    common colorbar.

    :param XY               :   The input coordinates
    :param dXYZ             :   The array that needed to be plotted
    :param unit_label       :   Label for the color bar (unit)
    :param contour          :   Boolean on whether contour line should be plotted
    :param plot             :   Boolean on whether to plot the figure
    :return                 :   Plot of the desired dXYZ with common colorbar
    """

    XY, dXYZ = np.array(XY), np.array(dXYZ)

    fig, axes = plt.subplots(nrows=dXYZ.shape[2], ncols=1, figsize=(8, 6))

    vmin = np.min(dXYZ)
    vmax = np.max(dXYZ)

    for i, ax in enumerate(axes):
        im = ax.pcolormesh(XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i], vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f'Component {i}')

        if contour:
            contour_levels = np.linspace(vmin, vmax, 10)
            ax.contour(XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i], levels=contour_levels, colors='k', linewidths=0.5)

    fig.subplots_adjust(hspace=0.5)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
    cbar.set_label(f'{unit_label}', fontsize=11)
    fig.supylabel("Y Coordinate [mm]")
    fig.supxlabel("X Coordinate [mm]")

    if plot:
        plt.show()
