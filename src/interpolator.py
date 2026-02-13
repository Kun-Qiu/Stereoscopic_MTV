import numpy as np
from scipy.interpolate import griddata

from utility.Visualization import plot_interpolation


class DisplacementInterpolator:
    def __init__(self, grid: np.ndarray, values: np.ndarray, density: int=10, method: str='cubic'):
        assert density >= 1, "Grid density must be greater or equal to 1."
        self.__density = density
        method = method.lower()
        
        assert method in ["linear", "cubic", "nearest"], "Invalid interpolation method."
        self.__method = method

        self.__in_grid = np.array(grid)
        self.__in_values = np.array(values)
        
        self.__intp_grid, self.__intp_values = self.__interpolate_grid()


    def evaluate(self, pt: np.ndarray) -> np.ndarray:
        """
        Obtain the interpolated value for the point (xi, yi) given some known value
        of surrounding points.

        """
        point = np.array(pt)
        assert point.shape == (2,), "Single point must be a 2D coordinate."

        interpolated = np.array([
            griddata(self.__in_grid, self.__in_values[:, i], point, method=self.__method)
            for i in range(self.__in_values.shape[1])
            ])        
        return interpolated


    def __interpolate_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the entire displacement grid given the grid density and several known
        displacements.

        :return :   Interpolated map of the desired value
        """
        x_min, y_min = np.min(self.__in_grid, axis=0)
        x_max, y_max = np.max(self.__in_grid, axis=0)
        x_new = np.linspace(x_min, x_max, self.__density)
        y_new = np.linspace(y_min, y_max, self.__density)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        valid_mask = np.all(np.isfinite(self.__in_values), axis=1)
        interpolate_field = griddata(
            self.__in_grid[valid_mask], self.__in_values[valid_mask],
            (X_new, Y_new), method=self.__method
            )
        return np.dstack((X_new, Y_new)), interpolate_field


    def plot_interpolation(self, unit:str, contour:bool=False) -> None:
        """
        Plot the interpolation of the displacement using color plot

        :param unit         :   Label for the color plot
        :param contour      :   Boolean on whether contour lines should be plotted
        :return             :   None 
        """
        plot_interpolation(
            self.__intp_grid, self.__intp_values,
            unit=unit, contour=contour
            )
        return


    def get_interpolate(self) -> tuple:
        """
        Get the interpolated values for the coordinate and the displacement

        :return :   Interpolated grid and the associated displacement
        """
        return self.__intp_grid, self.__intp_values
