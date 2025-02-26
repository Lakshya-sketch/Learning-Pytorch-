# import torch
# tensorA = torch.tensor([[1,2,3],
#                        [4,5,6],
#                        [7,8,9],
#                        ])
# tensorB = torch.tensor([[10]])
# tensor = torch.stack([tensorA,tensorB])
# print(tensor)
# print("\n")
# tensor = tensor.squeeze()
# print(tensor)

import torch

# Create a tensor with singleton dimensions
tensor = torch.rand(1, 3, 1, 5)  # Shape: (1, 3, 1, 5)
print(tensor)
print("\n")

# Apply squeeze
squeezed_tensor = tensor.squeeze()  # Removes all dimensions of size 1
print(squeezed_tensor)

print("Original Shape:", tensor.shape)  # (1, 3, 1, 5)
print("Squeezed Shape:", squeezed_tensor.shape)  # (3, 5)
