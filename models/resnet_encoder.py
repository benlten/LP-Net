import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

import torchvision

class ResnetEncoder(pl.LightningModule):
    def __init__(
            self, 
            transform_module = torch.nn.Identity(),
            encoder_out_size = 2048,
    ):
        super().__init__()

        self.save_hyperparameters()

        model = torchvision.models.resnet50(
                pretrained = False, 
                # norm_layer=nn.Identity
        )
        model.fc = torch.nn.Linear(2048, encoder_out_size)

        self.model = nn.Sequential(transform_module, model)

    def forward(self, data):
        data = self.model(data)
        return [ data ]
