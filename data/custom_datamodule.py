import torch
import torch.utils.data
import pytorch_lightning as pl

import os

from torch.utils.data import DataLoader, Dataset

from data import CustomDataset

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size = 32, num_workers = 0, augmentations = [torch.nn.Identity()]):
        super().__init__()
        self.data_path = data_path

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.augmentations = augmentations

    def prepare_data(self):
        pass

    def setup(self, resampled = True, stage=None, shuffle = True, device = 'cpu'):
        self.shuffle = shuffle
        self.device = device
        print('Loading training dataset...')
        self.train_dataset = FacesDataset(
            data_path = os.path.join(self.data_path, 'train'),
            augment = self.augmentations[0]
        )

        print('Loading validation datasets...')
        self.val_datasets = [ 
            FacesDataset(
              data_path = os.path.join(self.data_path, 'valid'),
              augment = augment,
            ) for augment in self.augmentations]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor = 6 if self.num_workers > 0 else 2,
            )

    def val_dataloader(self):
        """
        return DataLoader(self.val_dataset, 
                num_workers=self.num_workers, 
                batch_size=self.batch_size, 
                persistent_workers=True
                )
        """
        
        return [
            DataLoader(val_dataset,
              num_workers=self.num_workers, 
              batch_size=self.batch_size, 
              persistent_workers=True if self.num_workers > 0 else False,
              pin_memory=True,
              prefetch_factor = 6 if self.num_workers > 0 else 2,
              ) 
            for val_dataset in self.val_datasets 
        ]

if __name__ == '__main__':
    datamodule = FacesDataModule(
            data_path = '/awsmount/faces/8_identities', 
            salience_path = '/awsmount/salience_maps/8_identities', 
            crop_size = 180, 
            max_rotation = 15, 
            lp = True, 
            lp_out_shape = (190, 165), 
            augmentation = 'salience',
            points = 8,
    )
    datamodule.prepare_data()
    datamodule.setup()
