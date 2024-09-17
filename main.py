import patch
patch.all()

import os
import sys
import logging
#import copy
# logging.basicConfig(level=logging.getLevelName(os.environ.get('LOGLEVEL', 'NOTSET')), stream=sys.stdout)
# logging.getLogger().info('initialized logging')
import yaml
import datetime

import torchvision
import torch
import pytorch_lightning as pl

import wandb

from config import Config
from data import get_data, CustomDataset, CustomDataModule
from transformations import get_pipeline, Pipeline, Replicate
from models import ResnetEncoder
from utils import get_subset_of_size, TimerPrintCallback,  HParamsSaveCallback

import callbacks
import self_supervised
import supervised

def main(config: Config):
    print('master port', os.environ.get('MASTER_PORT', ''))
    print('master addr', os.environ.get('MASTER_ADDR', ''))
    pl.seed_everything(42)

    print(config)
    identities = config.data.classes

    default_root_dir = os.path.join(
            config.data.output,
            'tensorboard',
            f'data_{config.data.name}',
            f'model_{config.model.name}',
            f'transform_{config.transformations.type}',
            config.name,
    )

    tensorboard_root_dir = default_root_dir
    csv_root_dir = os.path.join(
            config.data.output,
            'csv',
            f'data_{config.data.name}',
            f'model_{config.model.name}',
            f'transform_{config.transformations.type}',
            config.name,
    )
    wandb_root_dir = os.path.join(
            config.data.output,
            'wandb',
            f'data_{config.data.name}',
            f'model_{config.model.name}',
            f'transform_{config.transformations.type}',
            config.name,
    )
    os.makedirs(wandb_root_dir, exist_ok=True)
    os.makedirs(csv_root_dir, exist_ok=True)
    os.makedirs(tensorboard_root_dir, exist_ok=True)

    print('Output directory:', default_root_dir)

    train_pipeline, eval_pipeline, collate_fn = get_pipeline(config.transformations)

#    ll = 1
#    if 'aug' in config.name:
#        ll = 5


    train_loaders, all_val_loaders, all_test_loaders = get_data(
        config,
        train_pipeline,
        eval_pipeline,
        collate_fn,
        repetitions = config.data.repetitions
    )

    if config.model.name == 'simclr':
        model = self_supervised.SimCLR(
            encoder = ResnetEncoder(encoder_out_size = config.model.zdim),
            gpus = torch.cuda.device_count(),
            num_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size * config.model.accumulate_grad_batches,
            dataset = 'custom',
            learning_rate = config.training.initial_lr,
            hidden_mlp = config.model.zdim,
        )
    elif config.model.name == 'byol':
        model = self_supervised.BYOL(
            base_encoder = ResnetEncoder( encoder_out_size = config.model.zdim ),
            encoder_out_size = config.model.zdim,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'barlow_twins':
        model = self_supervised.BarlowTwins(
            encoder = ResnetEncoder(encoder_out_size = config.model.zdim),
            encoder_out_dim = config.model.zdim,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'barlow_twins_dimensional':
        model = self_supervised.DimensionalBarlowTwins(
            encoder = ResnetEncoder(encoder_out_size = config.model.zdim),
            encoder_out_dim = config.model.zdim,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'siamese_net':
        model = self_supervised.SiameseNet(
            encoder = ResnetEncoder(encoder_out_size = config.model.zdim),
            encoder_out_dim = config.model.zdim,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'supervised_small':
        model = supervised.Baseline(
            in_dim = config.model.zdim,
            num_classes = config.model.classes,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'supervised_big':
        model = supervised.BaselineBig(
            in_dim = config.model.zdim,
            num_classes = config.model.classes,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'resnet_small':
        model = supervised.ResnetSmall(
            num_classes = config.model.classes,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'resnet':
        model = supervised.Resnet(
            num_classes = config.model.classes,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
    elif config.model.name == 'vgg':
        model = supervised.VGG(
            num_classes = config.model.classes,
            num_training_samples = sum(len(i.dataset) for i in train_loaders),
            batch_size = config.training.batch_size,
            learning_rate = config.training.initial_lr,
        )
        
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=tensorboard_root_dir)
    csv_logger = pl.loggers.CSVLogger(save_dir=csv_root_dir)
    # wandb_logger = pl.loggers.WandbLogger(
    #     project="LPNet-self-supervised", 
    #     group=config.name,
    #     tags=[ f'data_{config.data.name}', f'model_{config.model.name}', f'transform_{config.transformations.type}' ],
    #     # log_model = 'all', 
    #     # offline=True,
    #     save_dir=wandb_root_dir, 
    # )

    loggers = [
        tb_logger,
        csv_logger,
        # wandb_logger,
    ]

    save_last = pl.callbacks.ModelCheckpoint(
            dirpath = os.path.join(default_root_dir), 
            # filename='last',
            save_last = True
    )

    save_last.last_model_path = os.path.join(default_root_dir, 'last.ckpt')

    plugins = [
        pl.plugins.environments.TorchElasticEnvironment()
    ]

    ind = 0
    for train_loader, val_loaders in zip(train_loaders, all_val_loaders):
        
        print("CURRENT NUMBER OF IDENTITIES", config.data.classes[ind])

        trainer = pl.Trainer(
        # plugins=plugins,
        default_root_dir = default_root_dir,
        logger=loggers,
        accelerator = 'gpu',
        num_sanity_val_steps = 0,
        devices = -1,
        num_nodes = config.compute.num_nodes,
        precision = 16,
        accumulate_grad_batches = config.model.accumulate_grad_batches,
        max_epochs = config.training.epochs*(ind+1),
        log_every_n_steps = config.compute.logging_interval,
        strategy=pl.strategies.DDPStrategy(timeout=datetime.timedelta(minutes=5), find_unused_parameters=True, cluster_environment=plugins[0]),
        overfit_batches = config.compute.overfit_batches,
        callbacks = [
            pl.callbacks.ModelCheckpoint(monitor = 'val_loss'),
            save_last,
            HParamsSaveCallback(config),
#            callbacks.SSLOnlineEvaluator(
#                z_dim = config.model.zdim,
#                num_classes = config.data.classes[ind],
#                hidden_dim = None,
#                dataset = 'custom'
#            ),
#            callbacks.SampleImage(),
#            pl.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
            ],
        )


        print('Starting fit')
        trainer.fit_loop.current_epoch = config.training.epochs*ind
        ckpt_path = os.path.join(default_root_dir, 'last.ckpt')

        print("ORIGINAL CONFIG", config)

        # no checkpoint, this is a first run
        if not os.path.isfile(ckpt_path):
            ckpt_path = None
            print('not resuming, no checkpoint')
        else:
            conf = torch.load(ckpt_path)
            print("CHECKPOINT CONFIG", conf['full_config'])
            # config doesn't match, so this is a new run
            if config != conf['full_config']:
                ckpt_path = None
                print('not resuming, config does not match')
            
            elif conf['epoch'] >= config.training.epochs*(ind+1):
                print('config has run for', conf['epoch'], 'epochs, skipping to the next train_loader')
                ind+=1
                continue
            # config matches but has been completed, this is a new run
            elif conf['epoch'] == conf['full_config'].training.epochs * len(config.data.classes):
                ckpt_path = None
                print('not resuming, config matches but has been completed, this is a new run')
        if ckpt_path != None:
            print('going to resume')
            
            
        
            
        print("LENGTH OF TRAIN LOADER",len(train_loader))
        print("LENGTH OF VAL LOADER", len(val_loaders))

#        model_before = copy.deepcopy(model)
    
    
    
        for name, param in model.named_parameters():
            if(name == 'encoder.model.1.layer4.2.bn3.weight' ):
                print(param.data)
                
        print("trainer settings BEFORE : max and current epochs, loss", trainer.max_epochs, trainer.fit_loop.current_epoch, trainer.logged_metrics)
        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loaders, ckpt_path=ckpt_path)
        print("trainer settings BEFORE : max and current epochs, loss", trainer.max_epochs, trainer.fit_loop.current_epoch, trainer.logged_metrics)
        
        for name, param in model.named_parameters():
            if(name == 'encoder.model.1.layer4.2.bn3.weight' ):
                print(param.data)

                
#        model_after = copy.deepcopy(model)

#        weights_same = True
#        for p1, p2 in zip(model_before.parameters(), model_after.parameters()):
#            if p1.data.ne(p2.data).sum() > 0:
#                weights_same = False
#
#        if weights_same:
#            print("Weights are same")
#        else:
#            print("Weights are different")
            
        ind+=1

if __name__  ==  '__main__':
    # torch.set_float32_matmul_precision('medium')

    config = 'config.yaml'
    if config == '-':
        config = yaml.load(sys.stdin, Loader = yaml.SafeLoader)
    else:
        config = yaml.load(open(config, 'r'), Loader = yaml.SafeLoader)

    configs_to_run = [Config(**c) for c in config['runs']]
    if len(sys.argv) >  1:
        try:
            index = int(sys.argv[1])
            configs_to_run = [configs_to_run[index]]
        except:
            print('failed to parse argv')

    for run in configs_to_run:
        main(run)
