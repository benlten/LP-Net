import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from models.util import *

class OldModel(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity = x

        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn1(out)

        out += identity
        out = self.relu(out)

        out = self.avgpool(out)

        out = self.flat(out)
        out = self.fc(out)

        return out
