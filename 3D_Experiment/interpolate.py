import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def plot_interpolation(XY, dXYZ, unit_label):
    """
    Plot the given dXYZ array whether it is 1D, 2D, 3D with the associated
    common colorbar.

    :param XY               :   The input coordinates
    :param dXYZ             :   The array that needed to be plotted
    :param unit_label       :   Label for the color bar (unit)
    :return                 :   Plot of the desired dXYZ with common colorbar
    """

    XY, dXYZ = np.array(XY), np.array(dXYZ)

    fig, axes = plt.subplots(nrows=dXYZ.shape[2], ncols=1, figsize=(8, 6))

    vmin = np.min(dXYZ)
    vmax = np.max(dXYZ)

    for i, ax in enumerate(axes):
        im = ax.pcolormesh(XY[:, :, 0], XY[:, :, 1], dXYZ[:, :, i], vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f'Component {i}')

    fig.subplots_adjust(hspace=0.5)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location='right')
    cbar.set_label(f'{unit_label}', fontsize=11)
    fig.supylabel("Y Coordinate [mm]")
    fig.supxlabel("X Coordinate [mm]")
    plt.show()


class DisplacementInterpolator:
    def __init__(self, XY, dXY, grid_density=100, method='cubic', kx=3, ky=3, s=0):
        assert grid_density >= 1, "Desired grid density is smaller than original density."
        assert method.lower() == "linear" or "cubic" or "nearest"

        self.__points = np.array(XY)
        self.__displacement = np.array(dXY)
        self.__density = grid_density
        self.__kx = kx
        self.__ky = ky
        self.__s = s
        self.__method = method

        self.__interpolate_displacement = None
        self.__interpolate_grid = None
        self.compute_interpolate_grid()

    def compute_interpolate_point(self, XY):
        """
        Obtain the interpolated value for the point (xi, yi) given some known value
        of surrounding points.

        :param XY   :   The point whose dXY is needed
        :return     :   The displacement at that point is returned
        """
        point = np.array(XY)
        assert point.shape == (2,), "Single point must be a 2D coordinate."

        interpolated_x = griddata(self.__points, self.__displacement[:, 0], point, method=self.__method)
        interpolated_y = griddata(self.__points, self.__displacement[:, 1], point, method=self.__method)

        return np.array([interpolated_x, interpolated_y])

    def compute_interpolate_grid(self):
        """
        Interpolate the entire displacement grid given the grid density and several known
        displacements.

        :return :   Interpolated map of the desired value
        """
        x_min, y_min = np.min(self.__points, axis=0)
        x_max, y_max = np.max(self.__points, axis=0)
        x_new = np.linspace(x_min, x_max, self.__density)
        y_new = np.linspace(y_min, y_max, self.__density)

        X_new, Y_new = np.meshgrid(x_new, y_new)
        interpolate_vector = griddata(self.__points, self.__displacement,
                                      (X_new, Y_new), method=self.__method)

        self.__interpolate_displacement = interpolate_vector
        self.__interpolate_grid = np.dstack((X_new, Y_new))

    def plot_interpolation(self, unit_label):
        """
        Plot the interpolation of the displacement using color plot

        :param unit_label   :   Label for the color plot
        :return             :   None --> Plot
        """
        plot_interpolation(self.__interpolate_grid, self.__interpolate_displacement,
                           unit_label=unit_label)

    def get_interpolate(self):
        """
        Get the interpolated values for the coordinate and the displacement

        :return :   Interpolated grid and the associated displacement
        """
        return self.__interpolate_grid, self.__interpolate_displacement


# Example usage
if __name__ == "__main__":
    # Example Usage
    x_coords = np.array([394.1424255371094, 394.06536865234375, 394.0715637207031, 394.0648193359375,
                         394.068603515625, 394.14312744140625, 446.3945617675781, 446.57440185546875,
                         446.581787109375, 446.5845642089844, 446.574951171875, 446.4103088378906,
                         496.98883056640625, 496.6874084472656, 496.6806640625, 496.68194580078125,
                         496.6742248535156, 496.9674987792969])
    y_coords = np.array([280.7763977050781, 369.5273132324219, 459.23089599609375, 547.7984619140625,
                         637.4832763671875, 726.259765625, 286.4818420410156, 373.21728515625,
                         460.12994384765625, 546.9229125976562, 633.833740234375, 720.5584716796875,
                         291.7296142578125, 376.4134826660156, 461.1775207519531, 545.8704833984375,
                         630.62646484375, 715.3019409179688])
    x_displacement = np.array([1.0, 0.0, -1.0, 0.0,
                               0.5, -0.5, 1.0, -1.0,
                               0.8, -0.8, 0.5, -0.5,
                               0.2, -0.2, 0.3, -0.3,
                               0.4, -0.4])
    y_displacement = np.array([0.0, 1.0, 0.0, -1.0,
                               0.2, -0.2, 0.3, -0.3,
                               0.4, -0.4, 0.5, -0.5,
                               0.6, -0.6, 0.7, -0.7,
                               0.8, -0.8])

    points = np.column_stack((x_coords, y_coords))
    displacement = np.column_stack((x_displacement, y_displacement))
    interpolator = DisplacementInterpolator(points, displacement, grid_density=500)
    point = interpolator.compute_interpolate_point(np.array((400, 400)))
