from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning import strategies
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.classification import multiclass_accuracy

from self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        dataset: Optional[str] = None,
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None
        self.num_classes: Optional[int] = num_classes
        self.dataset: Optional[str] = dataset

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes
        if self.dataset is None:
            self.dataset = trainer.datamodule.name

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        if accel.is_distributed:
            if isinstance(accel.strategy, strategies.ddp.DDPStrategy):
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            #Can do a better fix
            # the state dict from the callback state only has module.block_forward.2.weight and module.block_forward.2.bias
            # the other information about the epochs is in optimizer state
            A = self._recovered_callback_state["state_dict"]
            B = self.online_evaluator.state_dict()

            equal = True
            for k1, k2 in zip(A,B):
                if(A[k1].size() != B[k2].size()):
                    equal = False
                    break
            
            if(equal):
                print("here")
                self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])

            if(equal == False):
                self._recovered_callback_state["optimizer_state"]["state"][1]['exp_avg'] = torch.zeros(self.num_classes)
                self._recovered_callback_state["optimizer_state"]["state"][1]['exp_avg_sq'] = torch.zeros(self.num_classes)
                p = self._recovered_callback_state["optimizer_state"]["state"][0]['exp_avg'].shape[1]
                self._recovered_callback_state["optimizer_state"]["state"][0]['exp_avg'] = torch.zeros((self.num_classes, p))
                self._recovered_callback_state["optimizer_state"]["state"][0]['exp_avg_sq'] = torch.zeros((self.num_classes, p))
                
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == "stl10":
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module(x).flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        acc = multiclass_accuracy(mlp_logits.softmax(-1), y, num_classes=self.num_classes, average='micro')

        return acc, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        train_acc, mlp_loss = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        val_acc, mlp_loss = self.shared_step(pl_module, batch)
        pl_module.log("online_val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def state_dict(self) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._recovered_callback_state = state_dict


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
