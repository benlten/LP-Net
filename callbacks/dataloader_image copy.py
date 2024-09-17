from typing import Any, Dict, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.decomposition import PCA
import io
from PIL import Image
import torchvision
import torchvision.transforms as transforms

class CustomLoggingCallback(Callback):
    def __init__(self, smoothing_factor=0.9):
        super().__init__()
        self.smoothing_factor = smoothing_factor
        self.running_loss = 0.0

    def on_batch_end(self, trainer, pl_module):
        # Log train loss at every batch end
        batch_loss = trainer.batch_loss
        self.running_loss = self.smoothing_factor * self.running_loss + (1 - self.smoothing_factor) * batch_loss
        trainer.logger.experiment.add_scalar("train_loss_batch", batch_loss, global_step=trainer.global_step)

    def on_epoch_end(self, trainer, pl_module):
        # Log train loss at every epoch end
        epoch_loss = trainer.callback_metrics['train_loss_epoch']
        trainer.logger.experiment.add_scalar("train_loss_epoch", epoch_loss, global_step=trainer.global_step)

        # Reset running_loss for the next epoch
        self.running_loss = 0.0

        # Calculate and log moving average after every epoch
        moving_average_loss = self.running_loss / (1 - self.smoothing_factor ** (trainer.global_step / trainer.batch_idx))
        trainer.logger.experiment.add_scalar("moving_average_loss", moving_average_loss, global_step=trainer.global_step)



class PlotLoss(Callback):  # pragma: no cover

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
    
    
  
