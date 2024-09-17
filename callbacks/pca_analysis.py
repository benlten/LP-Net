from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

class PCAAnalysis(Callback):  # pragma: no cover
    """
    Computes PCA on the output at each epoch. 

    Logs it as an image to logger[0]
    """

    def __init__(self, K = 1, select = True):
        self.selected_indices  = None
        if(select):
            self.selected_indices = torch.randint(0, 128, (20,))
        self.K = K

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.outputs = []
        self.targets = []

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
            x = batch[0][-1]
            x = x.to(pl_module.device)                
            representations = pl_module(x).flatten(start_dim=1)

            y = batch[1]

            self.outputs.append(representations.cpu())
            self.targets.append(y.cpu())

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        
        combined_outputs = torch.cat(self.outputs)
        combined_targets = torch.cat(self.targets)
        
        if self.selected_indices is None:
            X = combined_outputs.numpy()
            y = combined_targets.numpy()
        else:
            sel = torch.tensor(self.selected_indices)
            boo = torch.isin(combined_targets, sel)
            idx = torch.where(boo)[0]
#            print("indix", idx)
            X = combined_outputs[idx].numpy()
            y = combined_targets[idx].numpy()

#        print("SAVING EMBEDD", X.shape, y.shape )
        pl_module.loggers[0].experiment.add_embedding(mat=X, metadata=y, global_step=trainer.global_step)

