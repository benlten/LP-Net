import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from models.util import *

class SmallArchitecture(pl.LightningModule):
    def  __init__(self, num_classes, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size = 7, stride = 2, padding = 3, bias = False ),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),
                nn.Flatten(),
                nn.Linear(11552, 40),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(40, num_classes),
                )
