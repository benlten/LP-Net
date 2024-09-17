import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from model.resblock import ResBlock
from model.lstm import LSTM
from model.bilstm import BiLSTM

from model.util import *

class LargeArchitecture(pl.LightningModule):
    def __init__(self, num_classes, out_shape, n_crops = 0, final_layer_type='bilstm', **kwargs):
        super().__init__()

        if out_shape == (190, 165):
            self.in_size = 14720
        elif out_shape == (180, 180):
            self.in_size = 15488
        else:
            self.in_size = 21120

        if n_crops:
            self.model = nn.Sequential(
                    nn.Conv2d(3, hidden_dim, kernel_size = 7, stride = 2, padding = 3, bias = False ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding=1, bias = False),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
                    nn.Flatten(),
                    nn.Linear(self.in_size, 40), 
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    )

            if final_layer_type == 'lstm':
                self.final_layer = nn.LSTM(40, num_classes, batch_first=True)
            elif final_layer_type == 'bilstm':
                self.final_layer = BiLSTM(num_classes)
            else:
                self.final_layer = nn.Linear(40, num_classes)

            self.n_crops = n_crops

        else:
            self.model = nn.Sequential(
                    nn.Conv2d(3, hidden_dim, kernel_size = 7, stride = 2, padding = 3, bias = False ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, bias = False),
                    nn.ReLU(),
                    nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
                    nn.Flatten(),
                    nn.Linear(self.in_size, 40), 
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(40, num_classes),
                    )
            self.final_layer = None

    def forward(self, data):
        output = self.model(data)

        if self.n_crops:
            output_chunks = torch.split(output, self.n_crops, dim=0)
            if isinstance(self.final_layer, nn.Linear):
                final_layer_input = torch.stack([torch.mean(o, dim=0) for o in output_chunks], dim=0)
                output = self.final_layer(final_layer_input)
            elif isinstance(self.final_layer, nn.LSTM):
                final_layer_input = torch.stack(output_chunks)
                output = self.final_layer(final_layer_input)[1][0].squeeze()
            elif isinstance(self.final_layer, BiLSTM):
                final_layer_input = torch.stack(output_chunks)
                output = self.final_layer(final_layer_input)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch

        data = data.flatten(0,1)
        target = target.flatten()

        output = self.forward(data)

        loss = F.cross_entropy(output, target)
        # self.log('loss/train', loss.item(), prog_bar = True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        data = data.flatten(0,1)
        target = target.flatten()

        output = self.forward(data)

        loss = F.cross_entropy(output, target)
        self.log('loss/valid', loss.item(), prog_bar = True)

        return loss

