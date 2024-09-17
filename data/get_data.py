import torch
import os
from config import Config
from data import CustomDataset, CustomDataModule
import torchvision

def get_data(config: Config, train_pipeline, eval_pipeline, collate_fn, drop_last = False, repetitions=1):
    identities = config.data.classes

    train_loaders, all_val_loaders, all_test_loaders = [], [], []

    for identity in identities:
        data_path = os.path.join(config.data.dataset, config.data.name, f"{identity}_identities",)
        #changing this for now for continual, see how to merge with others later
        find_classes_data_path = os.path.join(config.data.dataset, config.data.name,
          f"{identity}_identities")

        print('Reading data from ', data_path)

        train_dataset = CustomDataset(
                f'{data_path}/train',
                augment = train_pipeline,
                find_classes_data_path = find_classes_data_path,
                repetitions = repetitions
        )
        val_dataset_1 = CustomDataset(
                f'{data_path}/valid',
                augment = eval_pipeline,
                find_classes_data_path = find_classes_data_path,
                repetitions = repetitions
        )
        test_dataset_1 = CustomDataset(
                f'{data_path}/test',
                augment = eval_pipeline,
                find_classes_data_path = find_classes_data_path,
                repetitions = repetitions
        )
#        train_dataset = torchvision.datasets.ImageNet(root=".", train=True, download=True, transform=train_pipeline)
#        val_dataset_1 = torchvision.datasets.ImageNet(root=".", train=False, download=True, transform=eval_pipeline)
#        test_dataset_1 = CustomDataset(
#                f'{data_path}/test',
#                augment = eval_pipeline,
#                find_classes_data_path = find_classes_data_path
#        )
        

        train_loader, val_loader_1, test_loader_1 = [
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.training.batch_size,
                    shuffle = index == 0, 
                    pin_memory=True,
                    num_workers=config.compute.num_workers,
                    persistent_workers=True if config.compute.num_workers else None,
                    # prefetch_factor=config.compute.num_workers if config.compute.num_workers else None,
                    collate_fn = collate_fn,
                    drop_last = drop_last
                )
                for index, dataset in enumerate([train_dataset, val_dataset_1, test_dataset_1])
        ]

        train_loaders.append(train_loader)
        all_val_loaders.append([ 
            val_loader_1 
        ])
        all_test_loaders.append([ 
            test_loader_1 
        ])

    return train_loaders, all_val_loaders, all_test_loaders

def get_datasets(config: Config, identity, train_pipeline, eval_pipeline, collate_fn):
    if config.data.name == 'DUMMY' and config.data.dataset == 'DUMMY':
        pass

    data_path = os.path.join(config.data.dataset, config.data.name, f"{identity}_identities",)
    find_classes_data_path = os.path.join(config.data.dataset, config.data.name, 
      f"{config.data.find_classes_data_path}_identities")

    print('Reading data from ', data_path)

    train_dataset = CustomDataset(
            f'{data_path}/train', 
            augment = train_pipeline,
            find_classes_data_path = find_classes_data_path
    )
    val_dataset_1 = CustomDataset(
            f'{data_path}/valid', 
            augment = eval_pipeline,
            find_classes_data_path = find_classes_data_path
    )
    test_dataset_1 = CustomDataset(
            f'{data_path}/test', 
            augment = eval_pipeline,
            find_classes_data_path = find_classes_data_path
    )
    return train_dataset, val_dataset_1, test_dataset_1
