import itertools
import torch.optim as optim
import OpticalFlow
import numpy as np
import cv2
import Optimization as op
import torch
from Utility import Visualization as vis
import matplotlib.pyplot as plt


def rmse(tensor_observed, tensor_truth):
    magnitude_observed = torch.norm(tensor_observed, dim=2)
    magnitude_truth = torch.norm(tensor_truth, dim=2)
    difference = magnitude_observed - magnitude_truth
    rmse_magnitude = torch.sqrt(torch.mean(difference ** 2))
    return rmse_magnitude


# Possible combinations
learning_rate_values = [1e-1, 1e-2, 1e-3]
lambda_smooth_values = np.arange(100, 250, 20)
lambda_intensity_values = np.arange(5, 50, 5)
lambda_vel_values = np.arange(25, 100, 10)

# Generate all possible combinations of parameters
param_combinations = list(
    itertools.product(learning_rate_values,
                      lambda_intensity_values,
                      lambda_vel_values,
                      lambda_smooth_values))
total_combination = len(param_combinations)

# ------------- Load Images and Template -------------------------------
source_path = '../Tool_2D/Experiment/SNR_4/Set_0/Gaussian_Grid_Image_Set_0.png'
target_path = '../Tool_2D/Experiment/SNR_4/Set_0/Rotational_Flow_Image_Set_0.png'
template_path = '../Tool_2D/Experiment/SNR_4/Template.png'
length_path = '../Tool_2D/length.txt'

source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)
template_image = cv2.imread(template_path)

of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()
of_flow = of_object.get_flow()

displacement = np.load('../Test/observed.npy').astype(int)
displacement_temp = []
for i in range(len(displacement[0])):
    displacement_temp.append((tuple(displacement[0][i]), displacement[1][i][0], displacement[1][i][1]))
truth_path = '../Tool_2D/Experiment/SNR_4/Set_0/Rotated_Field_Set_0.npy'
soa = rmse(torch.tensor(of_flow), torch.tensor(np.load(truth_path)))


# ----------------------- Optimization ---------------------------------------------------------------

def evaluate_parameters(learning_rate, lambda_smooth, lambda_int_grad, lambda_vel):
    model = op.DisplacementFieldModel(of_flow)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
    target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
    optimized_displacement = op.optimize_displacement_field(model, source_image_tensor,
                                                            target_image_tensor,
                                                            displacement_temp, optimizer,
                                                            lambda_smooth=lambda_smooth,
                                                            lambda_int_grad=lambda_int_grad,
                                                            lambda_vel=lambda_vel,
                                                            num_epochs=1000)
    return optimized_displacement


learning_rate = 5e-3
lambda_smooth = 100
lambda_vel = 25
lambda_int_grad = 5

df = evaluate_parameters(learning_rate, lambda_smooth, lambda_int_grad, lambda_vel)
vis.visualize_displacement(source_image, "Smoothed Optimized Displacement", df)
vis.visualize_displacement(source_image, "Initial Displacement", of_flow)
plt.show()
rmse_op = rmse(torch.tensor(df), torch.tensor(np.load(truth_path)))
print(rmse_op, soa)

# Perform grid search over all parameter combinations
# soa = rmse(torch.tensor(of_flow), torch.tensor(np.load(truth_path)))
# best_evaluation_metric = 100000000
# best_parameters = None
# cur_comb = 0
# for params in param_combinations:
#     learning_rate, lambda_intensity, lambda_vel, lambda_smooth = params
#     print(f"Current param: {params}")
#     displacement_op = evaluate_parameters(learning_rate, lambda_smooth, lambda_intensity, lambda_vel)
#     rmse_op = rmse(displacement_op, torch.tensor(np.load(truth_path)))
#     if rmse_op < best_evaluation_metric:
#         best_evaluation_metric = rmse_op
#         best_parameters = params
#         print(f"Optimal RMSE: {rmse_op}")
#     cur_comb = cur_comb + 1
#
# print("Optical Flow:", soa)
# print("Best parameters:", best_parameters)
# print("Best evaluation metric:", best_evaluation_metric)
