import torch.nn as nn

from config import (CONV1_PARAMS, CONV2_PARAMS, CONV3_PARAMS, 
                   CONV4_PARAMS, FC1_PARAMS, FC2_PARAMS)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv1 = nn.Conv2d(
            in_channels=CONV1_PARAMS['in_channels'],
            out_channels=CONV1_PARAMS['out_channels'],
            kernel_size=CONV1_PARAMS['kernel_size'],
            stride=CONV1_PARAMS['stride']
        )

        self.relu1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv2 = nn.Conv2d(
            in_channels=CONV2_PARAMS['in_channels'],
            out_channels=CONV2_PARAMS['out_channels'],
            kernel_size=CONV2_PARAMS['kernel_size'],
            stride=CONV2_PARAMS['stride']
        )

        self.relu2 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv3 = nn.Conv2d(
            in_channels=CONV3_PARAMS['in_channels'],
            out_channels=CONV3_PARAMS['out_channels'],
            kernel_size=CONV3_PARAMS['kernel_size'],
            stride=CONV3_PARAMS['stride']
        )

        self.relu3 = nn.ReLU(inplace=True)  

        # self.conv4 = nn.Conv2d(128, 64, 3, 1)
        self.conv4 = nn.Conv2d(
            in_channels=CONV4_PARAMS['in_channels'],
            out_channels=CONV4_PARAMS['out_channels'],
            kernel_size=CONV4_PARAMS['kernel_size'],
            stride=CONV4_PARAMS['stride']
        )

        self.relu4 = nn.ReLU(inplace=True)

        # self.fc1 = nn.Linear(256, 512)
        self.fc1 = nn.Linear(
            in_features=FC1_PARAMS['in_features'],
            out_features=FC1_PARAMS['out_features']
        )
        self.relu5 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(512, 2)   2 - number of actions
        self.fc2 = nn.Linear(
            in_features=FC2_PARAMS['in_features'],
            out_features=FC2_PARAMS['out_features']
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.fc2(out)

        return out
