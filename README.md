3D MTV Reconstruction with Soloff Polynomial Calibration
=======================================================

Overview
--------
This project implements a stereoscopic Molecular Tagging Velocimetry (MTV) system for 3D velocity field reconstruction using Soloff polynomial camera calibration.

Soloff Polynomial Calibration
----------------------------
Key Features:
- Implements Soloff polynomial mapping for stereoscopic camera calibration
- Handles lens distortion and perspective effects
- Supports multi-plane calibration targets
- Provides accurate 3D coordinate reconstruction

Calibration Process:
1. Calibration Target Setup:
   - Planar target with precisely spaced markers
   - Images acquired at multiple Z-positions (min 3 planes)

2. Polynomial Formulation:
   For each camera:
   x = Σ(a_ijk X^i Y^j Z^k)
   y = Σ(b_ijk X^i Y^j Z^k)
   where:
   - (x,y) = image coordinates
   - (X,Y,Z) = world coordinates
   - a_ijk, b_ijk = polynomial coefficients
   - Typically 3rd order (i+j+k ≤ 3)

3. Coefficient Calculation:
   - Detect target markers in calibration images
   - Solve for coefficients using least squares
   - Validate with reprojection error analysis

System Requirements
------------------
Software:
- Python 3.7+
- OpenCV, NumPy, SciPy, Matplotlib

Hardware:
- Two synchronized cameras
- Calibration target with known geometry
- Adequate lighting

Usage
-----
1. Calibration:
   python calibrate.py --left_images left_calib/ --right_images right_calib/ --target target_spec.yaml

2. 3D Reconstruction:
   python reconstruct.py --left_piv left_vectors.csv --right_piv right_vectors.csv --calib calibration_results.npz

Output
------
- Calibration coefficients (.npz format)
- 3D velocity fields (.vtk format)
- Validation plots

References
----------
[1] Soloff, S. M., et al. (1997). Measurement Science and Technology.