from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torchvision
from torch import nn
from torch.nn import functional as F
import torch

from pytorch_lightning import LightningModule


from utils import MemoryMonitor
from torchmetrics import Accuracy
import timm


class LogPolarNet(pl.LightningModule):
    def __init__(self, output_shape,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=1000):
        super(LogPolarNet, self).__init__()
        
        # Define encoder (ResNet-18)
        self.encoder = models.resnet18(pretrained=False)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        # Define deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=num_ftrs + 2, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)
        
        self.output_shape = output_shape

    def forward(self, x, coordinate):
        # Feature extraction
        features = self.encoder(x)
        
        # Concatenate with coordinate
        features = torch.cat((features, coordinate.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])), dim=1)
        
        # Deconvolutional layers
        x = F.relu(self.deconv1(features))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)
        x = F.interpolate(x, size=(self.output_shape[0], self.output_shape[1]), mode='bilinear', align_corners=False)
        
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, coord = batch
        y_pred = self(x[0], coord)
        loss = F.mse_loss(y_pred, y)
        self.log("epoch/train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, coord = batch
        y_pred = self(x[0], coord)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

