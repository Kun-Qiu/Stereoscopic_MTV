import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class DisplacementInterpolator:
    def __init__(self, points, displacement, grid_density=100, method='cubic', kx=3, ky=3, s=0):
        self.__points = np.array(points)
        self.__displacement = np.array(displacement)
        self.__density = grid_density
        self.__kx = kx
        self.__ky = ky
        self.__s = s
        self.__method = method
        self.__interpolate_displacement = None
        self.__interpolate_grid = None
        # self.norm = np.sqrt(np.sum(self.__displacement**2, axis=1))

        assert grid_density >= 1, "Desired grid density is smaller than original density."
        assert method.lower() == "linear" or "cubic" or "nearest"

    def interpolate(self):
        x_new = np.linspace(np.min(self.__points[:, 0]), np.max(self.__points[:, 0]), self.__density)
        y_new = np.linspace(np.min(self.__points[:, 1]), np.max(self.__points[:, 1]), self.__density)
        X_new, Y_new = np.meshgrid(x_new, y_new)

        interpolate_x_vector = griddata(self.__points, self.__displacement[:, 0],
                                        (X_new, Y_new),
                                        method=self.__method)

        interpolate_y_vector = griddata(self.__points, self.__displacement[:, 1],
                                        (X_new, Y_new),
                                        method=self.__method)

        self.__interpolate_displacement = (interpolate_x_vector, interpolate_y_vector)
        self.__interpolate_grid = (X_new, Y_new)

    def plot_interpolation(self, display_vector=False):
        if self.__interpolate_displacement is None:
            self.interpolate()

        x, y = self.__interpolate_grid
        x_hat, y_hat = self.__interpolate_displacement

        # Calculate magnitudes of unit vectors
        norm = np.sqrt(x_hat ** 2 + y_hat ** 2)
        unit_x = x_hat / norm
        unit_y = y_hat / norm

        plt.figure(figsize=(12, 6))

        # Plot scattered data with colored quiver plot
        plt.subplot(121)
        plt.scatter(self.__points[:, 0], self.__points[:, 1],
                    c=np.sqrt(np.sum(self.__displacement ** 2, axis=1)), cmap='viridis')
        plt.title('Scattered Data')

        plt.subplot(122)
        # if display_vector:
        #     plt.quiver(x[::step, ::step], y[::step, ::step],
        #                unit_x[::step, ::step], unit_y[::step, ::step], cmap='viridis', scale=50)
        plt.imshow(norm, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), origin='lower', cmap='viridis',
                   aspect='auto')
        plt.colorbar(label='Magnitude')
        plt.title('Interpolated Grid')
        plt.tight_layout()
        plt.show()

    def get_interpolate(self):
        if self.__interpolate_displacement is None:
            self.interpolate()

        x, y = self.__interpolate_grid
        x_hat, y_hat = self.__interpolate_displacement

        return x, y, x_hat, y_hat


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

    # Create an instance of DisplacementInterpolator
    interpolator = DisplacementInterpolator(points, displacement, grid_density=50)

    # Plot the interpolation
    interpolator.plot_interpolation()
