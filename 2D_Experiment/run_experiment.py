import cv2
import OpticalFlow
import TemplateMatching
import torch
import os

from Optimization import optimize_displacement_field
from Utility import Visualization as vis
from Utility import Template as tp

import numpy as np
import Optimization as op
import torch.optim as optim


def convert_tensor_numpy(input_tensor):
    """
    Convert PyTorch tensor to numpy array
    :param input_tensor:    Input torch tensor
    :return:                Numpy array
    """
    if torch.is_tensor(input_tensor):
        if input_tensor.requires_grad:
            return input_tensor.detach().numpy()
        else:
            return input_tensor.numpy()
    return input_tensor


def cosine_similarity(vec1, vec2):
    """
    Cosine similarity between two vectors of shape (1 , n)
    :param vec1:    First vector
    :param vec2:    Second vector
    :return:        Similarity (scalar) between two vectors
    """
    vec1 = convert_tensor_numpy(vec1)
    vec2 = convert_tensor_numpy(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def average_filter(input_tensor, window_size):
    """
    The average filter method can be used to filter out the vector maps by arithmetic
    averaging over vector neighbors to reduce the noise

    :param input_tensor:        Array of displacement vectors
    :param window_size:         The size of the neighborhood
    :return:                    Smoothed displacement field based on the neighborhood average
    """
    rows, cols, depth = input_tensor.shape
    assert depth == 2, "The input tensor must have a depth of 2."

    averaged_tensor = input_tensor.clone()
    half_window = window_size // 2

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
                averaged_tensor[i, j] = mean_vector

    return averaged_tensor


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


def test_optimization_function(source_path, target_path, template_path,
                               thresh_source, thresh_target, path, type):
    # Optical Flow
    of_object = OpticalFlow.OpticalFlow(source_path, target_path)
    of_object.calculate_optical_flow()
    of_field = of_object.get_flow()

    # Template Matching
    template_object = TemplateMatching.TemplateMatcher(source_path, target_path, template_path,
                                                       thresh_source=thresh_source,
                                                       thresh_target=thresh_target)
    template_object.match_template_driver()

    predicted = of_object.get_flow()
    observed = template_object.get_displacement()

    # ----------------------- Optimization ---------------------------------------------------------------
    # Initialize displacement field model and optimizer
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    model = op.DisplacementFieldModel(predicted)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    source_image_tensor = torch.tensor(source_image, dtype=torch.float32, requires_grad=True)
    target_image_tensor = torch.tensor(target_image, dtype=torch.float32, requires_grad=True)
    optimized_displacement = optimize_displacement_field(model, source_image_tensor,
                                                         target_image_tensor,
                                                         observed, optimizer,
                                                         lambda_smooth=100,
                                                         lambda_vel=25,
                                                         lambda_int_grad=5,
                                                         num_epochs=10000)

    field_filtered = moving_average_filter(optimized_displacement, window_size=8, threshold=0.8)
    field_averaged = average_filter(field_filtered, window_size=8)

    smoothed_displace_path = os.path.join(set_path, f"Smoothed_Optimized_Field_{type}.png")
    vis.visualize_displacement(source_image, "Smoothed Optimized Displacement", field_averaged,
                               save_path=smoothed_displace_path)

    initial_path = os.path.join(set_path, f"Optical_Field_{type}.png")
    vis.visualize_displacement(source_image, "Initial Displacement", predicted,
                               save_path=initial_path)

    displacement_diff_path = os.path.join(set_path, f"Difference_Field_{type}.png")
    vis.visualize_displacement_difference(optimized_displacement, predicted, source_image,
                                          save_path=displacement_diff_path)
    # plt.show()

    return field_averaged, of_field


def rmse(tensor_observed, tensor_truth):
    magnitude_observed = torch.norm(tensor_observed, dim=2)
    magnitude_truth = torch.norm(tensor_truth, dim=2)
    difference = magnitude_observed - magnitude_truth
    rmse_magnitude = torch.sqrt(torch.mean(difference ** 2))
    return rmse_magnitude


# ------------------------------- Experimental Code ---------------------------------------------
experiment_path = "../2D_Experiment/Experiment"
snr_values = [1, 2, 4, 8, 16]
set_range = 5

index_snr = 0
for value in snr_values:
    snr_path = os.path.join(experiment_path, f"SNR_{value}")
    for i in range(set_range):
        set_path = os.path.join(snr_path, f"Set_{i}")
        source_path = os.path.join(set_path, f"Gaussian_Grid_Image_Set_{i}.png")
        template_path = os.path.join(snr_path, f"Template.png")

        if not os.path.exists(template_path):
            template = tp.Template(source_path, template_path)
            template.run()

        # ----------------- Translational Image ------------------------------------
        print("Translational Flow")
        translate_path = os.path.join(set_path, f"Translational_Flow_Image_Set_{i}.png")
        trans_displace, of_trans = test_optimization_function(source_path, translate_path, template_path,
                                                              thresh_source=0.5,
                                                              thresh_target=0.5,
                                                              path=set_path,
                                                              type="Translational")
        trans_truth_displace = os.path.join(set_path, f"Translated_Field_Set_{i}.npy")
        rmse_trans = rmse(trans_displace, torch.tensor(np.load(trans_truth_displace)))
        rmse_of_trans = rmse(torch.tensor(of_trans), torch.tensor(np.load(trans_truth_displace)))

        # ------------------ Rotational Image ---------------------------------------
        print("Rotational Flow")
        rotate_path = os.path.join(set_path, f"Rotational_Flow_Image_Set_{i}.png")
        rotate_displace, of_rotate = test_optimization_function(source_path, rotate_path, template_path,
                                                                thresh_source=0.5,
                                                                thresh_target=0.5,
                                                                path=set_path,
                                                                type = 'Rotational')
        rotate_truth_displace = os.path.join(set_path, f"Rotated_Field_Set_{i}.npy")
        rmse_rotate = rmse(rotate_displace, torch.tensor(np.load(rotate_truth_displace)))
        rmse_of_rotate = rmse(torch.tensor(of_rotate), torch.tensor(np.load(rotate_truth_displace)))

        output_file_path = os.path.join(set_path, f"rmse_values_{i}.npy")
        rmse_values = {
            'rmse_translational': rmse_trans,
            'rmse_rotational': rmse_rotate,
            'rmse_of_translational': rmse_of_trans,
            'rmse_of_rotational': rmse_of_rotate
        }
        print(rmse_values)
        np.save(output_file_path, rmse_values)

    index_snr = index_snr + 1
