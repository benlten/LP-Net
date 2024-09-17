import torch
from torch import nn

class FinalLayerWrapper(nn.Module):
    def __init__(self, n_crops, model):
        super().__init__()
        self.n_crops = n_crops
        self.model = model

    def forward(self, x):
        return self.model(x.view(-1, self.n_crops, x.shape[-1]))
