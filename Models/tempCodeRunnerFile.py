import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CNN import train_func,test_func

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setup training data
train_data = datasets.FashionMNIST(
    root="E:\Pytorch", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="E:\Pytorch",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image , label = train_data[0]

class_names = train_data.classes

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

test_image = image[0]

# Create a convolutional neural network 
# class FashionMNISTModelV2(nn.Module):
#     """
#     Model architecture copying TinyVGG from: 
#     https://poloclub.github.io/cnn-explainer/
#     """
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
#         super().__init__()
#         self.block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape, 
#                       out_channels=hidden_units, 
#                       kernel_size=3, # how big is the square that's going over the image?
#                       stride=1, # default
#                       padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units, 
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2) # default stride value is same as kernel_size
#         )
#         self.block_2 = nn.Sequential(
#             nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             # Where did this in_features shape come from? 
#             # It's because each layer of our network compresses and changes the shape of our input data.
#             nn.Linear(in_features=hidden_units*7*7, 
#                       out_features=output_shape)
#         )
    
#     def forward(self, x: torch.Tensor):
#         x = self.block_1(x)
#         # print(x.shape)
#         x = self.block_2(x)
#         # print(x.shape)
#         x = self.classifier(x)
#         # print(x.shape)
#         return x

torch.manual_seed(42)
# model_2 = FashionMNISTModelV2(input_shape=1, 
#     hidden_units=10, 
#     output_shape=len(class_names)).to(device)
# print(model_2)

con_layer = nn.Conv2d(in_channels=1, 
                      out_channels=10, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 

print(con_layer.state_dict())
