import torch
import os
from config import Config
from data import CustomDataset, CustomDataModule

def get_eval_data(config: Config, pipelines):
    identities = config.data.classes

    train_loaders, all_val_loaders, all_test_loaders = [], [], []

    for identity in identities:
        data_path = os.path.join(config.data.dataset, config.data.name, f"{identity}_identities",)
        find_classes_data_path = os.path.join(config.data.dataset, config.data.name, 
          f"{config.data.find_classes_data_path}_identities")

        print('Reading data from ', data_path)

        for pipeline in pipelines:
            train_dataset = CustomDataset(
                    f'{data_path}/train', 
                    augment = pipeline,
                    find_classes_data_path = find_classes_data_path
            )
            val_dataset_1 = CustomDataset(
                    f'{data_path}/valid', 
                    augment = pipeline,
                    find_classes_data_path = find_classes_data_path
            )
            test_dataset_1 = CustomDataset(
                    f'{data_path}/test', 
                    augment = pipeline,
                    find_classes_data_path = find_classes_data_path
            )

            train_loader, val_loader_1, test_loader_1 = [
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.training.batch_size,
                    shuffle = False, 
                    pin_memory=True,
                    num_workers=config.compute.num_workers,
                )
                for dataset 
                in [train_dataset, val_dataset_1, test_dataset_1]
            ]

            train_loaders.append(train_loader)
            all_val_loaders.append(val_loader_1)
            all_test_loaders.append(test_loader_1)

    return train_loaders, all_val_loaders, all_test_loaders
