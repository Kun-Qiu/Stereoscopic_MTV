import itertools
import torch.optim as optim
import OpticalFlow
import TemplateMatching
import numpy as np
import cv2
from Utility.Template import correspondence_displacement
import Optimization as op
import torch
import torch.nn.functional as F

# Possible combinations
learning_rates = [1e-2]
lambda_intensity_values = np.arange(0, 100.0, 10)
lambda_vel_values = np.arange(0, 2000, 200)
num_epochs_values = [10000]

# Generate all possible combinations of parameters
param_combinations = list(
    itertools.product(learning_rates, lambda_intensity_values,
                      lambda_vel_values, num_epochs_values))
total_combination = len(param_combinations)

# ------------- Load Images and Template -------------------------------
source_path = 'Data/Source/frame_0.png'
target_path = 'Data/Target/synethetic_1.png'
template_path = 'Data/Template/frame_0_temp.png'

source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)
template_image = cv2.imread(template_path)

of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()

source = TemplateMatching
target = TemplateMatching
source.match_template()
target.match_template()

correspondence = source.matching_displacement(target)
observed = correspondence_displacement(correspondence)
predicted = of_object.get_flow()


# ----------------------- Optimization ---------------------------------------------------------------

def evaluate_parameters(learning_rate, lambda_intensity, lambda_vel, num_epochs):
    model = op.DisplacementFieldModel(predicted)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
    target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
    optimized_displacement = op.optimize_displacement_field(model, source_image_tensor,
                                                            target_image_tensor,
                                                            observed, optimizer,
                                                            lambda_int=lambda_intensity,
                                                            lambda_vel=lambda_vel,
                                                            num_epochs=num_epochs)
    # Return evaluation metric
    ground_truth = np.full((256, 256, 2), [3, 0])
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    cosine_similarities = F.cosine_similarity(optimized_displacement, ground_truth_tensor, dim=1)
    # Normalize cosine similarities to the range [0, 1] by adding 1 and dividing by 2
    component_similarity_normalized = (cosine_similarities + 1) / 2

    # Compute the average cosine similarity across all components
    avg_cosine_similarity = torch.mean(component_similarity_normalized)

    return avg_cosine_similarity.item()  # Return as Python float


# Perform grid search over all parameter combinations
best_evaluation_metric = 0
best_parameters = None
cur_comb = 0
for params in param_combinations:
    learning_rate, lambda_intensity, lambda_vel, num_epochs = params
    evaluation_metric = evaluate_parameters(learning_rate, lambda_intensity, lambda_vel, num_epochs)
    if evaluation_metric > best_evaluation_metric:
        best_evaluation_metric = evaluation_metric
        best_parameters = params
    print(f"Current iter: {cur_comb}, total iter: {total_combination}, param: {params}, cos_sim: {evaluation_metric}")
    cur_comb = cur_comb + 1

print("Best parameters:", best_parameters)
print("Best evaluation metric:", best_evaluation_metric)
