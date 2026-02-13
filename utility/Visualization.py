import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

    field = np.asarray(field)
    magnitudes = np.linalg.norm(field, axis=2)
    length, height, _ = image.shape

    plt.figure(figsize=(8, 6))
    if show_img:
        plt.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]], aspect='auto')

    if length == width:
        plt.quiver(range(0, length, step), range(0, height, step),
                   field[::step, ::step, 0], field[::step, ::step, 1],
                   magnitudes[::step, ::step],
                   angles='xy', scale_units='xy', scale=1, cmap='viridis')
    else:
        step_x = 9
        step_y = 16
        plt.quiver(range(0, height, step_y), range(0, length, step_x),
                   field[::step_x, ::step_y, 0], field[::step_x, ::step_y, 1],
                   magnitudes[::step_x, ::step_y],
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


def plot_interpolation(
    XY: np.ndarray, 
    dXYZ: np.ndarray,
    name: str,
    unit: str, 
    contour: bool = False, 
    path: str | None = None
) -> None:
    """
    Plot the given dXYZ array (2D/3D) with individual color scales for each component.
    Saves each component as a separate figure if path is provided, and also shows a combined figure.

    :param XY               : The input coordinates, shape (n, m, 2)
    :param dXYZ             : The array to be plotted, shape (n, m, n_components)
    :param unit       : Label for the color bar (unit)
    :param contour          : Boolean on whether contour lines should be plotted
    :param path             : Optional path to save individual component figures
    """
    XY, dXYZ = np.array(XY), np.array(dXYZ)
    n_components = dXYZ.shape[2]

    # Create combined figure with subplots
    fig_comb, axes_comb = plt.subplots(nrows=n_components, ncols=1, figsize=(8, 4*n_components))
    if n_components == 1:
        axes_comb = [axes_comb]

    for i, ax in enumerate(axes_comb):
        vmin = np.nanmin(dXYZ[:, :, i])
        vmax = np.nanmax(dXYZ[:, :, i])

        im = ax.pcolormesh(
            XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i],
            vmin=vmin, vmax=vmax,
            shading='auto'
        )
        ax.set_title(f'Component {i}')

        if contour:
            levels = np.linspace(vmin, vmax, 10)
            ax.contour(XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i], levels=levels, colors='k', linewidths=0.5)

        cbar = fig_comb.colorbar(im, ax=ax, location='right')
        cbar.set_label(name)

        # Save each component as its own figure if path provided
        if path is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            fig_single, ax_single = plt.subplots(figsize=(6, 5))
            im_single = ax_single.pcolormesh(
                XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i],
                vmin=vmin, vmax=vmax,
                shading='auto',
                cmap='RdBu_r'
                )
            
            if contour:
                ax_single.contour(XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i], levels=levels, colors='k', linewidths=0.5)
            
            ax_single.set_title(f'Component {i}', fontsize=14)
            cbar_single = fig_single.colorbar(im_single, ax=ax_single, location='right')
            cbar_single.set_label(f"{name} [{unit}]")
            fig_single.supxlabel(f"X Coordinate [mm]", fontsize=14)
            fig_single.supylabel(f"Y Coordinate [mm]", fontsize=14)
            fig_single.tight_layout()
            fig_single.savefig(Path(path) / f"{name}_component_{i}.png")
            plt.close(fig_single) 

    fig_comb.supxlabel("X Coordinate [mm]", fontsize=14)
    fig_comb.supylabel("Y Coordinate [mm]", fontsize=14)
    fig_comb.tight_layout()
    plt.show()
