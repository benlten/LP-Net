import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, kernel_size = 3):
        super().__init__()
        self.F = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size, stride = 1, padding = (kernel_size - 1) // 2), 
            nn.BatchNorm2d(hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size, stride = 1, padding = (kernel_size - 1) // 2), 
            nn.BatchNorm2d(out_dim), 
        )
        if (in_dim != out_dim):
            self.residual_transform = nn.Conv2d(in_dim, out_dim, kernel_size, stride = 1, padding = (kernel_size - 1)// 2)
        else:
            self.residual_transform = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
        assert self.F(torch.randn(1, in_dim, 100, 100)).shape == self.residual_transform(torch.randn(1, in_dim, 100, 100)).shape
        assert self.F(torch.randn(1, in_dim, 100, 100)).shape[1] == out_dim

    def forward(self, x):
        return self.relu(self.F(x) + self.residual_transform(x))
