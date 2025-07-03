import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

train_data = datasets.FashionMNIST(
    root="E:/Pytorch", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

test_data = datasets.FashionMNIST(
    root="E:/Pytorch", # where to download data to?
    train=False, # get test data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

class_names = train_data.classes

batch_size = 32

train_dataloader = DataLoader(
    train_data, # data
    batch_size=batch_size
) 

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size
)

class FashionMNIST(nn.Module):
    def __init__(self,input_features: int, hiddent_units: int, output_features: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels= input_features,
                out_channels= hiddent_units,
                kernel_size=3,
                stride=1
            ),
            nn.Relu(),
            nn.Conv2d(
                in_channels= input_features,
                out_channels= hiddent_units,
                kernel_size=3,
                stride=1
                ),
            nn.Relu(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=1
                )
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels= input_features,
                out_channels= hiddent_units,
                kernel_size=3,
                stride=1
                ),
            nn.Relu(),
            nn.Conv2d(
                in_channels= input_features,
                out_channels= hiddent_units,
                kernel_size=3,
                stride=1
                ),
            nn.ReLU()
            nn.MaxPool2d(
                kernel_size=3,
                stride=1
                )
        )

    def forward(self,x):
        x = self.block1
        print(f"Shape of X after Block 2: {x.shape}")