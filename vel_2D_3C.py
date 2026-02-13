import os
import numpy as np
import src.inverse_least_square as ils
import matplotlib.pyplot as plt

from src.interpolator import DisplacementInterpolator
from scipy.spatial import ConvexHull, Delaunay


def read_data(coeff_path):
    """
    Read the numpy file if a path is given, else if the input is a ndarray, return
    the ndarray

    :param coeff_path :   Unknown data (String or ndarray)
    :return           :   Return the content of the input_var
    """
    if isinstance(coeff_path, str) and os.path.exists(coeff_path):
        return np.load(coeff_path, allow_pickle=True).astype(float)
    elif isinstance(coeff_path, np.ndarray):
        return coeff_path.astype(float)


class Velocity2D_3C:
    def __init__(self, XY_l, XY_r, dXY_l, dXY_r, path_coeff_l, path_coeff_r):
        """
        Initialize the 2D 3C velocity calculator using Soloff calibration method
        """

        # Position of the reference points and the corresponding displacement
        self._XY_l = read_data(XY_l)    # Left Camera
        self._dXY_l = read_data(dXY_l)

        self._XY_r = read_data(XY_r)    # Right Camera
        self._dXY_r = read_data(dXY_r)

        # Load the calibration coefficient
        self._coeff_l = np.load(path_coeff_l, allow_pickle=True)
        self._coeff_r = np.load(path_coeff_r, allow_pickle=True)
        self._inverse_obj = ils.InverseTransform(self._coeff_l, self._coeff_r)

        # Interpolate the displacement to a grid
        self._left_intp = DisplacementInterpolator(XY_l, dXY_l, density=500)
        self._right_intp = DisplacementInterpolator(XY_r, dXY_r, density=500)

        self._build_hulls()

    
    def _build_hulls(self):
        """
        Build convex hulls and Delaunay triangulations for both cameras.
        """
        pts_l = self._XY_l.reshape(-1, 2)
        pts_r = self._XY_r.reshape(-1, 2)

        self._hull_l = ConvexHull(pts_l)
        self._hull_r = ConvexHull(pts_r)

        self._hull_l_delaunay = Delaunay(pts_l[self._hull_l.vertices])
        self._hull_r_delaunay = Delaunay(pts_r[self._hull_r.vertices])


    def __project_2D_delta(self, xyz:np.ndarray, name:str) -> np.ndarray:
        """
        Extract the 2D displacement at known 3D points
        """
        
        width, height, _ = xyz.shape
        dXY_2D = np.full((width, height, 2), np.nan)

        for i, column in enumerate(xyz):
            for j, point in enumerate(column):

                # Project a 3D point to 2D camera plane
                camera_pt = self._inverse_obj.projection_object_to_image(
                    point, name
                    )

                print(f"Projected 2D point for {name} camera: {camera_pt}")

                if name.lower() == "left":
                    # The projected point must be within the convex hull of the individual camera
                    if not self.points_in_left_hull(np.array(camera_pt)):
                        continue
                    displacement = self._left_intp.evaluate(camera_pt)
                elif name.lower() == "right":
                    if not self.points_in_right_hull(np.array(camera_pt)):
                        continue
                    displacement = self._right_intp.evaluate(camera_pt)
                else:
                    raise ValueError(
                        f"Camera name {name} is not recognized. Only 'left' or 'right' is allowed."
                        )
                
                dXY_2D[i, j] = displacement.ravel()
        return dXY_2D


    def interpolate_3D_displacement(self, XY:np.ndarray, dXYZ:np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the 3D displacement at several known point to a desired grid

        :param XY       :   Known coordinates
        :param dXYZ     :   Known displacement (3D)
        :return         :   Interpolated grid and displacement
        """
        assert dXYZ.shape[2] == 3, f"Input displacement matrix have dimension of {dXYZ.shape[2]}, but a dim of 3 " \
                                f"is required."
        if XY.shape[2] != 2:
            XY = XY[:, :, 0:2]

        intp_obj = DisplacementInterpolator(
            XY.reshape(-1, 2), dXYZ.reshape(-1, 3), 
            density=500
            )
        return intp_obj.get_interpolate()


    def calculate_3D_displacement(self, xyz: np.ndarray) -> np.ndarray:
        """
        Calculate the 3D displacement using Soloff Calibration method

        :param xyz  :   3D coordinate where displacement is desire
        :return     :   Array of 3D displacement at xyz
        """
        if not isinstance(xyz, np.ndarray) or xyz.ndim != 3:
            raise ValueError("xyz must be a 3D numpy array")
     
        # Project 3D Points onto 2D Image and retrieve the corresponding displacement 
        # [dim1, dim2] of xyz
        left_dXY = self.__project_2D_delta(xyz, 'left').reshape(-1, 2)     
        right_dXY = self.__project_2D_delta(xyz, 'right').reshape(-1, 2)   

        N = xyz.reshape(-1, 3).shape[0]
        displace_3D = np.full((N, 3), np.nan)
        for i, point in enumerate(xyz.reshape(-1, 3)):
            
            if np.any(np.isnan(left_dXY[i])) or np.any(np.isnan(right_dXY[i])):
                # Ensure point within convex hull of points
                continue

            # Inverse transformation to calculate the 3D displacement
            displace_3D[i] = self._inverse_obj.inverse_displacement(
                point,
                left_dXY[i],
                right_dXY[i]
                )
                
        return np.array(displace_3D).reshape(xyz.shape)
    
    
    def points_in_left_hull(self, points):
        """
        Check if points are inside the left-camera convex hull.
        """
        points = np.atleast_2d(points)
        return self._hull_l_delaunay.find_simplex(points) >= 0


    def points_in_right_hull(self, points):
        """
        Check if points are inside the right-camera convex hull.
        """
        points = np.atleast_2d(points)
        return self._hull_r_delaunay.find_simplex(points) >= 0
