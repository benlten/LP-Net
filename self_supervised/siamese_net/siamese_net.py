from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union

import torch
from torch import nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from self_supervised.lr_scheduler import LinearWarmupCosineAnnealingLR

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

class SiameseNet(LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        margin = 100,
        lambda_coeff=5e-3,
        z_dim=8192,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=1000,
    ):
        super().__init__()

        self.save_hyperparameters(ignore="encoder")

        self.encoder = encoder

        self.train_iters_per_epoch = num_training_samples // batch_size

        self.margin = margin


    def forward(self, x):
        return self.encoder(x)[0]

    def shared_step(self, batch):
        imgs, _ = batch
        imgs = torch.stack(imgs[:-1]) # K,N,C,H,W
        imgs = imgs.transpose(0,1)
        N,K,C,H,W = imgs.shape

        h = self.encoder(imgs.flatten(0,1))[0] # K*N, D
        # Z = self.projector(h)
        Z = h

        idx_range = torch.arange(len(Z)).reshape(-1,1) // K * K
        pos_indexes = idx_range + torch.arange(K)
        pos_tensors = Z[pos_indexes]
        pos_distance = (Z.unsqueeze(1) - pos_tensors).norm(dim=-1)
        corrected_pos_distance = 0.5 * (pos_distance**2)
        # pos_similarity = torch.nn.functional.cosine_similarity(Z.repeat_interleave(pos_tensors.shape[1], 0), pos_tensors.flatten(0,1)).mean()

        neg_indexes = torch.randint(0, len(Z)-K, (len(Z),K))
        # all_neg_indexes = torch.arange(0, len(Z) - K).repeat(len(Z),1)
        neg_indexes_corrected = (neg_indexes + idx_range + K) % len(Z)
        neg_tensors = Z[neg_indexes_corrected]
        neg_distance = (Z.unsqueeze(1) - neg_tensors).norm(dim=-1)
        corrected_neg_distance = 0.5*(torch.nn.functional.relu(self.margin-neg_distance) ** 2)
        # neg_similarity = torch.nn.functional.cosine_similarity(Z.repeat_interleave(neg_tensors.shape[1], 0), neg_tensors.flatten(0,1)).mean()

        return {
                'loss': corrected_neg_distance.mean() + corrected_pos_distance.mean(), 
                # 'pos_similarity': pos_similarity,
                'pos_distance': pos_distance.mean().detach(),
                # 'neg_similarity': neg_similarity,
                'neg_distance': neg_distance.mean().detach(),
                'output': Z.detach().cpu(),
        }

    def training_step(self, batch, batch_idx):
        d = self.shared_step(batch)
        self.log("loss/train", d['loss']**0.5, on_step=True, on_epoch=False)
        # self.log("pos_similarity/train", d['pos_similarity'], on_step=True, on_epoch=False)
        # self.log("neg_similarity/train", d['neg_similarity'], on_step=True, on_epoch=False)
        self.log("pos_distance/train", d['pos_distance'], on_step=True, on_epoch=False)
        self.log("neg_distance/train", d['neg_distance'], on_step=True, on_epoch=False)
        return d['loss']

    def validation_step(self, batch, batch_idx):
        X, y = batch
        K = len(X) - 1
        y = y.repeat_interleave(K).detach().cpu()

        d = self.shared_step(batch)
        self.log("val_loss", d['loss']**0.5, on_step=False, on_epoch=True)
        self.log("hp_metric", d['loss']**0.5, on_step=False, on_epoch=True)

        self.log("loss/val", d['loss']**0.5, on_step=True, on_epoch=False)
        # self.log("pos_similarity/val", d['pos_similarity'], on_step=True, on_epoch=False)
        # self.log("neg_similarity/val", d['neg_similarity'], on_step=True, on_epoch=False)
        self.log("pos_distance/val", d['pos_distance'], on_step=True, on_epoch=False)
        self.log("neg_distance/val", d['neg_distance'], on_step=True, on_epoch=False)

        return [ d['output'], y ]

    def validation_epoch_end(self, outputs):
        return

        # if outputs:
            # X = torch.cat([X for X,_ in outputs]).numpy()
            # y = torch.cat([y for _,y in outputs]).numpy()
            # Z = PCA(n_components=2).fit_transform(X)
            # plt.clf()
            # plt.scatter(x=Z[:,0], y=Z[:,1], c=y)
            # self.logger[0].experiment.add_figure('PCA_viz_outputs_2d', plt.gcf(), self.global_step)
            # plt.clf()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        return optimizer

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [scheduler]
