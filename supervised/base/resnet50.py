from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torchvision
from torch import nn
from torch.nn import functional as F
import torch

from pytorch_lightning import LightningModule

from supervised.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils import MemoryMonitor
from torchmetrics import Accuracy

class Resnet(LightningModule):
    def __init__(self, num_classes,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=1000,):
        
        super().__init__()
        
        self.save_hyperparameters()
        
        self.resnet_model = torchvision.models.resnet50(pretrained = False)
        self.model = nn.Sequential(*(list(self.resnet_model.children())[:-2]))
        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,num_classes)
      
        self.train_iters_per_epoch = num_training_samples // batch_size
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)
    

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        cer = nn.CrossEntropyLoss()
        
        logits = self(x[0])
        
        loss = cer(logits, y)
        accu = self.metric(logits, y)
        
        print("trainloss", loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_accu", accu, on_step=True, on_epoch=False)
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
