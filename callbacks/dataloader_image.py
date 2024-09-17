from typing import Any, Dict, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.decomposition import PCA
import io
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# Define mean and standard deviation used during normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create a denormalization transform
denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                   std=[1/s for s in std])

class SampleImage(Callback):  # pragma: no cover
    """
    Computes PCA on the output at each epoch.

    Logs it as an image to logger[0]
    """

    def __init__(self, K = 4, select = True):
        self.selected_indices  = None
        if(select):
            self.selected_indices = torch.randint(0, 10, (K,))
        self.K = K

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.train_img = []
        self.train_label = []

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.val_img = []
        self.val_label = []
        
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            # last input is for online eval
#             print(pl_module, len(batch[0]))#, batch[1].shape)
#             print(len(batch), len(batch[0]), batch[0][-1].shape)
            x = batch[0][-1][self.selected_indices]
            y = batch[1][self.selected_indices]
                
            x = denormalize(x)

#             x = x.to(pl_module.device)
#             print(x.shape)
        
            self.train_img.append(x.cpu())
            self.train_label.append(y.cpu())

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        with torch.no_grad():
            # last input is for online eval
#             print(pl_module, len(batch[0]))#, batch[1].shape)
#             print(len(batch), len(batch[0]), batch[0][-1].shape)
            x = batch[0][-1][self.selected_indices]
            y = batch[1][self.selected_indices]

#             x = x.to(pl_module.device)
#             print(x.shape)
            x = denormalize(x)
        
            self.val_img.append(x.cpu())
            self.val_label.append(y.cpu())
     
     
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        current_epoch = trainer.current_epoch
        idx = 0
#        print("size", len(self.train_img), self.train_img[0].shape)
        for i in self.train_img:
            idx +=1
            grid = torchvision.utils.make_grid(i, nrow=2)
            pl_module.loggers[0].experiment.add_image("train_dataloader/epoch_"+str(current_epoch)+"/batch_"+str(idx)+"/gs_"+str(trainer.global_step), grid, global_step=trainer.global_step)


    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        current_epoch = trainer.current_epoch
        idx = 0
#        print("size", len(self.val_img), self.val_img[0].shape)
        for i in self.val_img:
#            print(i)
            idx +=1
            grid = torchvision.utils.make_grid(i, nrow=2)
            pl_module.loggers[0].experiment.add_image("val_dataloader/epoch_"+str(current_epoch)+"/batch_"+str(idx)+"/gs_"+str(trainer.global_step), grid, global_step=trainer.global_step)
    
    
  
