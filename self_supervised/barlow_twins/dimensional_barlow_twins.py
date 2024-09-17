from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from self_supervised.lr_scheduler import LinearWarmupCosineAnnealingLR

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)

class DimensionalBarlowTwins(LightningModule):
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
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def shared_step(self, batch):
        imgs, _ = batch
        imgs = torch.stack(imgs[:-1]) # K,N,C,H,W

        Z = self.projector(self.encoder(imgs.flatten(0,1))[0]) # K*N, D
        Z = Z.reshape(imgs.shape[0], imgs.shape[1], -1) # K,N,D

        # minimize to reduce variance within K transformations of a single base image
        intra_image_variance = Z.std(dim=0).mean() # std along K dimension, mean along N,D dimensions

        # find mean of all K images from single base image, compute similarity between all N base image means, minimize similarities
        mean = Z.mean(dim=0) # N,D
        similarity = (mean / mean.norm(dim=1).unsqueeze(1)) @ (mean.T / mean.norm(dim=1).unsqueeze(0))
        inter_image_similarity = self.off_diagonal(similarity).mean()

        return intra_image_variance + inter_image_similarity

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
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
