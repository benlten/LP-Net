import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from models.resblock import ResBlock

from models.lstm import LSTM
from models.bilstm import BiLSTM
from models.averaging_linear import AveragingLinear

from models.final_layer_wrapper import FinalLayerWrapper

import torchvision

class Model(pl.LightningModule):
    def __init__(self, num_classes, out_shape, n_crops = 0, final_layer_type='bilstm', hidden_dim=64, depth=1, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        if out_shape == (190, 165):
            self.in_size = 2016 * hidden_dim
            self.in_size = 25 * hidden_dim
        elif out_shape == (180, 180):
            self.in_size = 2025 * hidden_dim
            self.in_size = 25 * hidden_dim
        else:
            self.in_size = 21120
        

        if kwargs.get('resnet', False):
            self.model = torchvision.models.resnet18()
            out_size = 512
            if n_crops:
                if final_layer_type == 'lstm':
                    final_layer = FinalLayerWrapper(n_crops, LSTM(num_classes, input_dim=out_size))
                elif final_layer_type == 'bilstm':
                    final_layer = FinalLayerWrapper(n_crops, BiLSTM(num_classes, input_dim=out_size))
                else:

                    final_layer = FinalLayerWrapper(n_crops, AveragingLinear(num_classes, input_dim=out_size))
            self.model.fc = final_layer
        

        else: 
            self.model = [ 
                    ResBlock(3, hidden_dim, hidden_dim),
                    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                    *[ResBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(depth)],
                    nn.AdaptiveAvgPool2d(5),
                    nn.Flatten(),
                    nn.Linear(self.in_size, 40), 
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    ]
            out_size = 40
            if n_crops:
                if final_layer_type == 'lstm':
                    final_layer = FinalLayerWrapper(n_crops, LSTM(num_classes, input_dim=out_size))
                elif final_layer_type == 'bilstm':
                    final_layer = FinalLayerWrapper(n_crops, BiLSTM(num_classes, input_dim=out_size))
                else:

                    final_layer = FinalLayerWrapper(n_crops, AveragingLinear(num_classes, input_dim=out_size))
            self.model = nn.Sequential(*self.model, final_layer)

        self.n_crops = n_crops

    def forward(self, data, dim=0):
        output = self.model(data)

        return output

    def training_step(self, batch, batch_idx):
        data, target = batch

        data = data.flatten(0,1)
        target = target.flatten()

        output = self.forward(data)

        loss = F.cross_entropy(output, target)
        self.log('loss/train', loss.item(), prog_bar = True)

        accuracy = (output.argmax(dim = -1) == target).float().mean()
        self.log('acc/train', accuracy.item(), prog_bar = True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        data, target = batch

        data = data.flatten(0,1)
        target = target.flatten()

        output = self.forward(data)

        loss = F.cross_entropy(output, target)
        accuracy = (output.argmax(dim = -1) == target).float().mean()
        if dataloader_idx == 0:
            self.log('loss/val_upright', loss.item(), prog_bar = True, sync_dist=True, add_dataloader_idx = False)
            self.log('acc/val_upright', accuracy.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)
        else:
            self.log('loss/val_inverted', loss.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)
            self.log('acc/val_inverted', accuracy.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.hparams.schedule_frequency, gamma = self.hparams.schedule_gamma)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler }

if __name__ == '__main__':
    # n, c, h, w

    X = torch.randn(24, 3, 190, 165)
    y = torch.randint(16, (24,))

    print('testing without crops')
    m = Model(num_classes = 16, out_shape = (190, 165), n_crops = 0, final_layer_type='bilstm')
    print(m(X).shape)
    assert(m(X).shape[0] == 24)
    assert(m(X).shape[1] == 16)

    m.training_step((X,y), 0)


    print('testing with 4-crops')
    m = Model(num_classes = 16, out_shape = (190, 165), n_crops = 4, final_layer_type='bilstm')
    print(m(X).shape)
    assert(m(X).shape[0] == 6)
    assert(m(X).shape[1] == 16)

    m.training_step((X,y), 0)
