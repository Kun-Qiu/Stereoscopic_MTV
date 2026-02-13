import os, csv
import numpy as np
import src.calibrate_least_square as nsl


def read_csv_points(csv_path):
    """
    Reads CSV with headers: Camera# X Y x y z
    """
    assert os.path.exists(csv_path), f"CSV path {csv_path} does not exist."
    
    points = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract columns as floats
            X = float(row['X'])
            Y = float(row['Y'])
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            points.append([X, Y, x, y, z])
    return np.array(points)


class CalibrationPointDetector:
    def __init__(self, left_csv_path, right_csv_path, save_path):
        assert os.path.exists(left_csv_path), f"Given path: {left_csv_path}, does not exist."
        assert os.path.exists(right_csv_path), f"Given path: {right_csv_path}, does not exist."
        
        self._save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        self._left_points = read_csv_points(left_csv_path)
        self._right_points = read_csv_points(right_csv_path)

        # The object plane position should be the same for both left and right camera
        self._left_calibrated_points = self._left_points[:, -3:]    # x, y, z
        self._right_calibrated_points = self._right_points[:, -3:]  # x, y, z
        self._left_image_points = self._left_points[:, 0:2]         # X_l, Y_l
        self._right_image_points = self._right_points[:, 0:2]       # X_r, Y_r
        assert len(self._left_calibrated_points) == len(self._right_calibrated_points) == len(self._left_image_points) == len(self._right_image_points), \
            "Length of calibration and distortion from either left or right camera are not equal."


    def run_calibration(self):
        """
        Main driver for the execution of the calibration
        """

        left_nsl_object = nsl.CalibrationTransformation(
            calibrated_points=self._left_calibrated_points,
            distorted_points=self._left_image_points,
            z_order=1
            )
        
        right_nsl_object = nsl.CalibrationTransformation(
            calibrated_points=self._right_calibrated_points,
            distorted_points=self._right_image_points,
            z_order=1
            )

        print("Calculating transformation coefficient for cameras...")
        left_nsl_object.calibrate_least_square()
        right_nsl_object.calibrate_least_square()

        print(f"Saving coefficient to the following path: {self._save_path}")
        left_nsl_object.save_calibration_coefficient(self._save_path, "left_cam_coeff")
        right_nsl_object.save_calibration_coefficient(self._save_path, "right_cam_coeff")

        print("Completed the calculation of transformation coefficients")

if __name__ == "__main__":
    """
    Calibration Coefficient for the LaVision Coordinates
    """
    left_csv = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\cam1_pts.csv"
    right_csv = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\cam2_pts.csv"
    save_path = r"C:\Users\Kun Qiu\Projects\Stereoscopic_MTV\experiment"
    detector = CalibrationPointDetector(left_csv, right_csv, save_path)
    detector.run_calibration()
