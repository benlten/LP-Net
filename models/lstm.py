import torch
from torch import nn
from torch.nn import functional as F

class LSTM(nn.Module):
    def __init__(self, num_classes, input_dim=40):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, num_classes, batch_first=True)
        self.linear = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        lstm_out = self.lstm(x)[1][0]
        lstm_out = lstm_out.permute(1,0,2).reshape(len(x), -1)
        return self.linear(lstm_out)
