import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def minimize_total_distance(set1, set2):
    # Ensure the sets are numpy arrays
    set1 = np.array(set1)
    set2 = np.array(set2)

    n, m = set1.shape[0], set2.shape[0]
    size = max(n, m)

    # Create the cost matrix with a large value for dummy points
    cost_matrix = np.full((size, size), np.inf)

    # Fill the cost matrix with actual distances
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = np.linalg.norm(set1[i] - set2[j])

    # Fill the cost matrix for dummy points to prevent infeasibility
    if n < size:
        for i in range(n, size):
            cost_matrix[i, :] = 0
    if m < size:
        for j in range(m, size):
            cost_matrix[:, j] = 0

    # Solve the linear sum assignment problem (minimize total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter out the dummy assignments and calculate total distance
    correspondence = [(row, col) for row, col in zip(row_ind, col_ind) if row < n and col < m]
    total_distance = sum(cost_matrix[row, col] for row, col in correspondence)

    return correspondence, total_distance


def visualize_correspondence(set1, set2, correspondence):
    set1 = np.array(set1)
    set2 = np.array(set2)

    plt.figure(figsize=(8, 8))
    plt.scatter(set1[:, 0], set1[:, 1], c='blue', label='Set 1')
    plt.scatter(set2[:, 0], set2[:, 1], c='red', label='Set 2')

    for (i, j) in correspondence:
        plt.plot([set1[i, 0], set2[j, 0]], [set1[i, 1], set2[j, 1]], 'k--')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Optimal Correspondence between Two Sets of Points')
    plt.grid(True)
    plt.show()


# Example usage
set1 = [(1, 2), (3, 4), (5, 6)]
set2 = [(5, 4), (3, 2)]

correspondence, total_distance = minimize_total_distance(set1, set2)
print(correspondence)
print("Correspondence between set1 and set2:", correspondence)
print("Total distance:", total_distance)

visualize_correspondence(set1, set2, correspondence)
