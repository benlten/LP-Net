from typing import Any, Dict, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.decomposition import PCA
import io
from PIL import Image
import torchvision

class PCAImageTrain(Callback):  # pragma: no cover
    """
    Computes PCA on the output at each epoch.

    Logs it as an image to logger[0]
    """

    def __init__(self, K = 20, select = False):
        self.selected_indices  = None
        if(select):
            self.selected_indices = torch.randint(0, 128, (K,))
        self.K = K
        self.pca = PCA(2)

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.outputs = []
        self.targets = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if(pl_module.current_epoch % 50 == 0):
            with torch.no_grad():
                # last input is for online eval
    #             print(pl_module, len(batch[0]))#, batch[1].shape)
                x = batch[0][-1]

                x = x.to(pl_module.device)
                representations = pl_module(x).flatten(start_dim=1)
                y = batch[1]
                print("pick one of the three images in the batch", x.shape, "then flatten", representations.shape, y.shape)


                self.outputs.append(representations.cpu())
                self.targets.append(y.cpu())

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        print(pl_module.current_epoch)
        if(pl_module.current_epoch % 50 == 0):
            combined_outputs = torch.cat(self.outputs)
            combined_targets = torch.cat(self.targets)

            if self.selected_indices is None:
    #             print("KK ", self.K * len(self.outputs), self.K, len(self.outputs))
    #             corrected_K = int(self.K * len(self.outputs))
    #             self.selected_indices = torch.randperm(len(self.outputs))[:corrected_K]
                X = combined_outputs.numpy()
                y = combined_targets.numpy()
            else:
                sel = torch.tensor(self.selected_indices)
                boo = torch.isin(combined_targets, sel)
                idx = torch.where(boo)[0]
                print("indix", idx)
                X = combined_outputs[idx].numpy()
                y = combined_targets[idx].numpy()

            X_pca = self.pca.fit_transform(X)
            plt.scatter(X_pca[:,0], X_pca[:,1], c =y ,  cmap='hsv')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
#             plt.show()
            print(pl_module.loggers[0].save_dir, pl_module.loggers[0].log_dir)
            plt.savefig(pl_module.loggers[0].log_dir + '/epoch_' + str(pl_module.current_epoch) +'.png', bbox_inches='tight', dpi=300)
            print("SAVING img" )
#             pl_module.loggers[0].experiment.add_figure("PCA", plt.gcf(), global_step=trainer.global_step, close=True)
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            im = Image.open(buf)
            im = torchvision.transforms.ToTensor()(im)

            pl_module.loggers[0].experiment.add_image("pca", im, global_step=trainer.global_step)
            plt.close()

