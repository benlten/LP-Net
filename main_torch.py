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
import torch.nn as nn
import numpy as np
from torchmetrics import Accuracy
#import pytorch_lightning as pl

# import wandb

from config import Config
from data import get_data, CustomDataset, CustomDataModule
from transformations import get_pipeline, Pipeline, Replicate
from models import ResnetEncoder
from supervised.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils import get_subset_of_size, TimerPrintCallback,  HParamsSaveCallback


import self_supervised
import supervised
from torch.utils.tensorboard import SummaryWriter



class Model(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity = x

        out = self.conv2(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn1(out)

        out += identity
        out = self.relu(out)

        out = self.avgpool(out)

        out = self.flat(out)
        out = self.fc(out)

        return out


def train(log_interval, model, device, train_loader, optimizer, epoch, metric, writer):
    model.train()
    losses = []
    accus = []
    for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.type(torch.cuda.FloatTensor)
        print("sizes", len(data), data[0].shape, target.shape)
        data = data[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        cer = nn.CrossEntropyLoss()
        loss = cer(output, target)
        temp_loss = loss.detach().cpu().numpy()
        accu = metric(output, target).detach().cpu().numpy()
        losses.append(temp_loss)
        accus.append(accu)
        writer.add_scalar("train_loss", temp_loss, epoch)
        writer.add_scalar("train_accu", accu, epoch)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item(), accu))

    losses = np.array(losses)
    average_loss = np.mean(losses)
    average_accu = np.mean(np.array(accus))
    
    return average_loss


def evalu(log_interval, model, device, val_loader, optimizer, epoch, metric, writer):
    model.eval()
    losses = []
    accus = []
    for batch_idx, (data, target) in enumerate(val_loader):
#         data = data.type(torch.cuda.FloatTensor)
        data = data[0]
        data, target = data.to(device), target.to(device)
        output = model(data)
        cer = nn.CrossEntropyLoss()
        loss = cer(output, target)
        temp_loss = loss.detach().cpu().numpy()
        accu = metric(output, target).detach().cpu().numpy()
        losses.append(temp_loss)
        accus.append(accu)
        writer.add_scalar("val_loss", temp_loss, epoch)
        writer.add_scalar("val_accu", accu, epoch)

        if batch_idx % log_interval == 0:
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx, len(val_loader),
                100. * batch_idx / len(val_loader), loss.item(), accu))

    losses = np.array(losses)
    average_loss = np.mean(losses)
    average_accu = np.mean(np.array(accus))
    return average_loss

def main(config: Config):
    print('master port', os.environ.get('MASTER_PORT', ''))
    print('master addr', os.environ.get('MASTER_ADDR', ''))


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

    train_loaders, all_val_loaders, all_test_loaders = get_data(
        config,
        train_pipeline,
        eval_pipeline,
        collate_fn,
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
    elif config.model.name == 'supervised':
        model = Model(
            in_dim = config.model.zdim,
            num_classes = config.model.classes
        ).to('cuda')
    
    
    print(model)
    ckpt_path = os.path.join(default_root_dir, 'last.ckpt')

    print("ORIGINAL CONFIG", config)
    start_epoch = 0

    # no checkpoint, this is a first run
    if not os.path.isfile(ckpt_path):
#        ckpt_path = None
        print('not resuming, no checkpoint')
    else:
        print('going to resume')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch  = checkpoint['epoch']
#    else:
#        conf = torch.load(ckpt_path)
#        print("CHECKPOINT CONFIG", conf['full_config'])
#        # config doesn't match, so this is a new run
#        if config != conf['full_config']:
##            ckpt_path = None
#            print('not resuming, config does not match')
#
#        elif conf['epoch'] == conf['full_config'].training.epochs * len(config.data.classes):
##            ckpt_path = None
#            print('not resuming, config matches but has been completed, this is a new run')
     
    
    train_loader = train_loaders[0]
    val_loader = all_val_loaders[0][0]
    optimizer =  torch.optim.Adam(list(model.parameters()), lr =  config.training.initial_lr) #check scheduling
    metric = Accuracy(task="multiclass", num_classes=config.model.classes).to('cuda')
    writer = SummaryWriter(tensorboard_root_dir)
    print(ckpt_path)
    
    for iteri in range(start_epoch,config.training.epochs):
        print(len(train_loader))
        train(10, model, 'cuda', train_loader, optimizer, iteri, metric, writer)
        print(len(val_loader))
        evalu(10, model, 'cuda', val_loader, optimizer, iteri, metric, writer)
        torch.save({
            'epoch': iteri,
            'model': model.state_dict(),
            }, ckpt_path)
    
    
    writer.close()
    
   

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
