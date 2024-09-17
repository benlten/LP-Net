from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from self_supervised.lr_scheduler import LinearWarmupCosineAnnealingLR

from self_supervised import ProjectionHead
from utils import MemoryMonitor

class BarlowTwins(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=8192,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=1000,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="encoder")

        self.encoder = encoder
        self.projector = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)[0]

    def off_diagonal(self, x): 
        #################### from https://github.com/facebookresearch/barlowtwins/blob/main/main.py ###########################
        ####################################################### begin reference ###############################################
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        ####################################################### end reference #################################################

    def shared_step(self, batch):
        imgs, _ = batch
        img1, img2 = imgs[:2]

        #################### from https://github.com/facebookresearch/barlowtwins/blob/main/main.py ###########################
        ####################################################### begin reference ###############################################
        z1 = self.projector(self.encoder(img1)[0])
        z2 = self.projector(self.encoder(img2)[0])

        # empirical cross-correlation matrix
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.hparams.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(cross_corr).pow_(2).sum()
        loss = on_diag + self.hparams.lambda_coeff * off_diag
        ####################################################### end reference #################################################
        return loss

    def training_step(self, batch, batch_idx):
        # print('model start shared', batch_idx, MemoryMonitor().str(), flush=True)
        loss = self.shared_step(batch)
        # print('model finish shared', batch_idx, MemoryMonitor().str(), flush=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

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
