import cv2
import OpticalFlow
import TemplateMatching
import matplotlib.pyplot as plt
from Utility import Visualization as vis
import Optimization as op
from Optimization import optimize_displacement_field
import torch
import numpy as np


def rmse(tensor_observed, tensor_truth):
    magnitude_observed = torch.norm(tensor_observed, dim=2)
    magnitude_truth = torch.norm(tensor_truth, dim=2)
    difference = magnitude_observed - magnitude_truth
    rmse_magnitude = torch.sqrt(torch.mean(difference ** 2))
    return rmse_magnitude


def convert_tensor_numpy(input_tensor):
    if torch.is_tensor(input_tensor):
        if input_tensor.requires_grad:
            return input_tensor.detach().numpy()
        else:
            return input_tensor.numpy()
    return input_tensor


def cosine_similarity(vec1, vec2):
    vec1 = convert_tensor_numpy(vec1)
    vec2 = convert_tensor_numpy(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def moving_average_filter(input_tensor, window_size, threshold):
    """
    Moving average filter which compares displacement vectors with its neighbors.
    If the deviation is above a threshold, the vector is replaced with the average
    of the neighborhood of window_size.

    :param input_tensor:    The displacement vectors with their spatial positions (list of tuples).
    :param window_size:     The size of the neighborhood
    :param threshold:       Threshold value for deviation from average of neighborhood.
    :return:                Array of validated displacement vectors based on moving average.
    """
    # Get dimensions of the input tensor
    rows, cols, depth = input_tensor.shape
    assert depth == 2, "The input tensor must have a depth of 2."

    # Define the output tensor
    output_tensor = input_tensor.clone()

    # Define the half window size
    half_window = window_size // 2

    # Iterate over each element in the input tensor
    for i in range(rows):
        for j in range(cols):
            current_vector = input_tensor[i, j]
            neighborhood_vectors = []

            # Collect the neighborhood vectors
            for m in range(max(0, i - half_window), min(rows, i + half_window + 1)):
                for n in range(max(0, j - half_window), min(cols, j + half_window + 1)):
                    if (m, n) != (i, j):  # Exclude the center vector itself
                        neighborhood_vectors.append(input_tensor[m, n])

            # Compute the mean of the neighborhood vectors
            if neighborhood_vectors:
                neighborhood_vectors = torch.stack(neighborhood_vectors)
                mean_vector = torch.mean(neighborhood_vectors, dim=0)

                # Compute the cosine similarity between the current vector and the mean vector
                sim = cosine_similarity(current_vector, mean_vector)

                # Replace the current vector with the mean vector if similarity is above threshold
                if sim < threshold:
                    output_tensor[i, j] = mean_vector

    return output_tensor


# Load original image and displacement field (example)
source_path = '../Tool_2D/Experiment/SNR_4/Set_0/Gaussian_Grid_Image_Set_0.png'
target_path = '../Tool_2D/Experiment/SNR_4/Set_0/Rotational_Flow_Image_Set_0.png'
template_path = '../Tool_2D/Experiment/SNR_4/Template.png'
length_path = '../Tool_2D/length.txt'

# Optical Flow
of_object = OpticalFlow.OpticalFlow(source_path, target_path)
of_object.calculate_optical_flow()
of_flow = of_object.get_flow()

# Template Matching
template_object = TemplateMatching.TemplateMatcher(source_path, target_path, template_path,
                                                   length_path)
template_object.match_template_driver()

predicted = of_object.get_flow()
observed = template_object.get_displacement()

# ----------------------- Optimization ---------------------------------------------------------------
# Initialize displacement field model and optimizer
source_image = cv2.imread(source_path)
target_image = cv2.imread(target_path)

model = op.DisplacementFieldModel(predicted)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
optimized_displacement = optimize_displacement_field(model, source_image_tensor,
                                                     target_image_tensor,
                                                     observed, optimizer,
                                                     lambda_smooth=100,
                                                     lambda_vel=25,
                                                     lambda_int_grad=5,
                                                     num_epochs=10000)

truth_path = '../Tool_2D/Experiment/SNR_4/Set_0/Rotated_Field_Set_0.npy'

# field_filtered = moving_average_filter(optimized_displacement, window_size=3, threshold=0.6)
# field_averaged = average_filter(optimized_displacement, window_size=8)
# vis.visualize_displacement(source_image, "Optimized Displacement", optimized_displacement)
vis.visualize_displacement(source_image, "Smoothed Optimized Displacement", optimized_displacement)
vis.visualize_displacement(source_image, "Initial Displacement", predicted)
rmse_op = rmse(optimized_displacement, torch.tensor(np.load(truth_path)))
rmse_of = rmse(torch.tensor(of_flow), torch.tensor(np.load(truth_path)))
print(f"rmse_op: {rmse_op}, rmse_of: {rmse_of}")
# vis.visualize_displacement_difference(optimized_displacement, predicted, source_image)
plt.show()
