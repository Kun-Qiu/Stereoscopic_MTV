import numpy as np
from scipy.optimize import least_squares


# Define the non-linear warp transformation model
def dewarp_model(params, xi, yi):
    a11, a12, a13, a14, a15, a16, a21, a22, a23, a24, a25, a26, a31, a32, a33, a34, a35, a36 = params
    denominator = (a31 * xi) + (a32 * yi) + a33 + (a34 * xi ** 2) + (a35 * yi ** 2) + (a36 * xi * yi)
    x_dewarped = ((a11 * xi) + (a12 * yi) + a13 + (a14 * xi ** 2) + (a15 * yi ** 2) + (a16 * xi * yi)) / denominator
    y_dewarped = ((a21 * xi) + (a22 * yi) + a23 + (a24 * xi ** 2) + (a25 * yi ** 2) + (a26 * xi * yi)) / denominator
    return x_dewarped, y_dewarped


# Objective function to minimize (residuals)
def residuals(params, xi, yi, x_true, y_true):
    x_pred, y_pred = dewarp_model(params, xi, yi)
    return np.concatenate([x_pred - x_true, y_pred - y_true])


num_points = 18
np.random.seed(0)  # For reproducibility

x = np.random.rand(num_points) * 10  # Original x positions (random values)
y = np.random.rand(num_points) * 10  # Original y positions (random values)

# Create observed positions with some random noise
noise_level = 0.1
x_observed = x + np.random.randn(num_points) * noise_level
y_observed = y + np.random.randn(num_points) * noise_level

# Initial guess for the parameters (18 parameters for the model)
initial_params = np.ones(18)

# Perform the optimization using Levenberg-Marquardt algorithm
result = least_squares(residuals, initial_params, args=(x, y, x_observed, y_observed), method='lm')

# Optimized parameters
optimized_params = result.x

print("Optimized parameters:", optimized_params)
