# model = Net()

'''
NYU Computer Vision CS 480

** Usage **
Initialize a model:
    net = Net().to(device)

Load a model from pth file:
    net = Net().to(device)
    net_state_dict = torch.load(PATH_TO_NET)
    net.load_state_dict(net_state_dict,strict=False)

** Citation **
The implementation of the following Net is based on the following paper but with modifications (the local contrast normalization is not used):
Deep Neural Network for Traffic Sign Recognition Systems: An analysis of Spatial Transformers and Stochastic Optimization Methods
(Alvaro Arcos-Garc ́ıa, Juan A. Alvarez-Garc ́ıa, Luis M. Soria-Morillo)
Paper Link: https://idus.us.es/bitstream/handle/11441/80679/NEUNET-D-17-00381.pdf

The implementation of the Spatial Transformer Networks (STN) is based on the following tutorial:
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
'''

import torch
import torchvision

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models

import time
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from glob import glob
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=200,kernel_size=(7,7),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=200,out_channels=250,kernel_size=(4,4),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=250, out_channels=350,kernel_size=(4,4),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(350 * 6 * 6, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 43),
        )

        # Spatial transformer localization-network
        self.localization_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=3,out_channels=250,kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=250,out_channels=250, kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )

        self.localization_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=200,out_channels=150,kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=150,out_channels=200, kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )

        self.localization_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=250,out_channels=150,kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(in_channels=150,out_channels=200, kernel_size=(5,5),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
        )

        # Regressors for the affine matrix
        self.fc_loc_1 = nn.Sequential(
            nn.Linear(250 * 6 * 6, 250),
            nn.ReLU(True),
            nn.Linear(250, 6),
        )

        self.fc_loc_2 = nn.Sequential(
            nn.Linear(200 * 2 * 2, 300),
            nn.ReLU(True),
            nn.Linear(300, 6),
        )

        self.fc_loc_3 = nn.Sequential(
            nn.Linear(200 * 1 * 1, 300),
            nn.ReLU(True),
            nn.Linear(300, 6),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc_1[2].weight.data.zero_()
        self.fc_loc_1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc_2[2].weight.data.zero_()
        self.fc_loc_2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc_3[2].weight.data.zero_()
        self.fc_loc_3[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, localization, fc_loc, flat_size):
        xs = localization(x)
        xs = xs.view(-1, flat_size)
        theta = fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,x):
        # Spatial transformer network
        x = self.stn(x,self.localization_1,self.fc_loc_1,250*6*6)
        # Conv layers
        x = self.conv_layer1(x)
        # Spatial transformer network
        x = self.stn(x,self.localization_2,self.fc_loc_2,200*2*2)
        # Conv layers
        x = self.conv_layer2(x)
        # Spatial transformer network
        x = self.stn(x,self.localization_3,self.fc_loc_3,200*1*1)
        # Conv layers
        x = self.conv_layer3(x)
        # flatten
        x = x.view(-1,350 * 6 * 6)
        # FC layer
        x = self.fc_layer(x)
        return x
