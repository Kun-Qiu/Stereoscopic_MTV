import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Convert numpy arrays to PyTorch variables
# source_img = Variable(torch.from_numpy(source_img).float(), requires_grad=False)
# target_img = Variable(torch.from_numpy(target_img).float(), requires_grad=False)
# flow = Variable(torch.from_numpy(flow).float(), requires_grad=True)
#
# # Define optimization parameters
# learning_rate = 0.1
# num_iterations = 100
#
#
# # Define the energy functional to minimize
# def energy_functional(source_img, target_img, flow):
#     # Reshape the flow tensor to match the expected size for the grid tensor
#     flow = flow.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension and change dimensions order
#     shifted_source_img = F.grid_sample(source_img.unsqueeze(0), flow, align_corners=True)
#     return torch.sum((shifted_source_img.squeeze(0) - target_img) ** 2)
#
#
# # Define optimizer
# optimizer = torch.optim.SGD([flow], lr=learning_rate)
#
# # Perform optimization
# for i in range(num_iterations):
#     # Zero gradients
#     optimizer.zero_grad()
#
#     # Calculate loss
#     loss = energy_functional(source_img, target_img, flow)
#
#     # Backpropagation
#     loss.backward()
#
#     # Update flow field
#     optimizer.step()
#
# # Extract displacement field from the optimized flow
# dx = flow[:, :, 0].detach().numpy()
# dy = flow[:, :, 1].detach().numpy()
#
# # Create quiver plot
# plt.figure()
# plt.imshow(target_img.numpy(), cmap='gray')
# plt.quiver(range(0, target_img.shape[1], 10),
#            range(0, target_img.shape[0], 10),
#            dx[::10, ::10],
#            dy[::10, ::10],
#            color='r', scale=10)
# plt.show()