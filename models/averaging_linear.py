import torch
from torch import nn
from torch.nn import functional as F

class AveragingLinear(nn.Module):
    def __init__(self, num_classes, input_dim=40):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x.mean(dim=1))

