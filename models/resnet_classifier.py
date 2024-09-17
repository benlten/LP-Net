import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

import torchvision

class ResnetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.save_hyperparameters()

        self.model = torchvision.models.resnet50(pretrained = False)
        self.model.fc = torch.nn.Sequential(
            nn.Linear(2048,1000),
            nn.Linear(1000,num_classes),
        )

    def forward(self, data, dim=0):
        output = self.model(data)

        return output

    def shared_step(self, batch, batch_idx):
        data, target = batch

        output = self.forward(data)

        loss = F.cross_entropy(output, target)
        accuracy = (output.argmax(dim = -1) == target).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.shared_step(batch, batch_idx)

        self.log('loss/train', loss.item(), prog_bar = True)
        self.log('acc/train', accuracy.item(), prog_bar = True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, accuracy = self.shared_step(batch, batch_idx)

        if dataloader_idx == 0:
            self.log('loss/val_upright', loss.item(), prog_bar = True, sync_dist=True, add_dataloader_idx = False)
            self.log('acc/val_upright', accuracy.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)
        else:
            self.log('loss/val_inverted', loss.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)
            self.log('acc/val_inverted', accuracy.item(), prog_bar = False, sync_dist=True, add_dataloader_idx = False)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3, weight_decay = 1e-3)
        return optimizer

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.hparams.schedule_frequency, gamma = self.hparams.schedule_gamma)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler }
