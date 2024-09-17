from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch

import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from supervised.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils import MemoryMonitor
from torchvision.utils import make_grid
from torchmetrics import Accuracy

class Baseline(LightningModule):
    def __init__(self, in_dim, num_classes,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=1000,):
        
        super().__init__()
        
        self.save_hyperparameters()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)
        
        self.train_iters_per_epoch = num_training_samples // batch_size
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.random_indices = torch.randint(0, 128, (4,))
        
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
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        cer = nn.CrossEntropyLoss()
        
        logits = self(x[0])
        
        loss = cer(logits, y)
        accu = self.metric(logits, y)
        
        print("trainloss", loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_accu", accu, on_step=True, on_epoch=False)
        self.loggers[0].experiment.add_histogram("train/x", x[0], global_step=self.global_step)
        self.loggers[0].experiment.add_histogram("train/y", y, global_step=self.global_step)
        # Use the same set of random indices for both training and validation steps
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        cer = nn.CrossEntropyLoss()
        
        logits = self(x[0])
        
        loss = cer(logits, y)
        accu = self.metric(logits, y)
        
        print("valloss", loss, accu)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_accu", accu, on_step=False, on_epoch=True)
        self.loggers[0].experiment.add_histogram("val/x", x[0], global_step=self.global_step)
        self.loggers[0].experiment.add_histogram("val/y", y, global_step=self.global_step)
        
        return loss
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )

        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         linear_warmup_decay(warmup_steps),
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        return [optimizer], [scheduler]
