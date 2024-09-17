import patch
patch.all()

import os
import time
import sys
import logging
# logging.basicConfig(level=logging.getLevelName(os.environ.get('LOGLEVEL', 'NOTSET')), stream=sys.stdout)
# logging.getLogger().info('initialized logging')
import yaml
import datetime
import tqdm as tqdm

import torchvision
import torch
import pytorch_lightning as pl

import numpy as np
import pandas as pd

from config import Config
from data import get_eval_data
from transformations import get_eval_pipeline
from utils import get_subset_of_size, TimerPrintCallback,  HParamsSaveCallback

import callbacks
import supervised
import prepare_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DOWNLOADED = set()

def main(ckpt_index, ckpt_path, TRAIN_PERCENTAGE = 1.00):

    if not os.path.isfile(ckpt_path):
        print('no checkpoint')
        return
    
    print('loading checkpoint')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print('checkpoint loaded')

    try:
        config = ckpt['full_config']

#        if 'model' not in list(ckpt['state_dict'].keys())[0].split('.'):
#            backbone = pl_bolts_resnet50()
#        elif '1' in list(ckpt['state_dict'].keys())[0].split('.'):
#            backbone = ResnetEncoder(encoder_out_size = 2048)
#        else:
#            backbone = OldResnetEncoder(encoder_out_size = 2048)


        if config.model.name == 'supervised_small':
            model = supervised.Baseline.load_from_checkpoint(
                ckpt_path,
            )
        elif config.model.name == 'supervised_big':
            model = supervised.BaselineBig.load_from_checkpoint(
                ckpt_path,
            )
        elif config.model.name == 'resnet_small':
            model = supervised.ResnetSmall.load_from_checkpoint(
                ckpt_path,
            )
        elif config.model.name == 'resnet':
            model = supervised.Resnet.load_from_checkpoint(
                ckpt_path,
            )
        elif config.model.name == 'vgg':
            model = supervised.VGG.load_from_checkpoint(
                ckpt_path,
            )
        else:
            return

    except Exception as err:
        print(err)
        print('BROKEN MODEL: ', ckpt_path)
        return

    model = model.to(device)
    model.eval()

    if config.data.name not in DOWNLOADED:
        prepare_data.extract_aws2(config.data.name, config.data.dataset)
        DOWNLOADED.add(config.data.name)

    pl.seed_everything(42)

#    config.training.batch_size = 256
    config.compute.num_workers = 8

    print(config)
    pipelines = get_eval_pipeline(config.transformations)

    train_loaders, val_loaders, test_loaders = get_eval_data(
        config, 
        pipelines,
    )


#    print('fitting linear classifier on upright training data')
#    X = []
#    y = []
#    with torch.no_grad():
#        for index, (imgs, labels) in enumerate(train_loaders[0]):
#            imgs = imgs.to(model.device)
#            X.append(model(imgs).cpu())
#            y.append(labels.cpu())
#
#    X = torch.cat(X).to(model.device)
#    y = torch.cat(y).to(model.device)
#
#    indices = torch.randperm(len(X))
#    amt_to_train = int(len(X) * TRAIN_PERCENTAGE)
#    indices = indices[:amt_to_train]
#    X_subset, y_subset = X[indices], y[indices]
#
#    clf = torch.nn.Linear(X.shape[1], y.max() + 1).to(model.device)
#    optimizer = torch.optim.Adam(clf.parameters())
#
#    print('initial loss, acc:',
#            torch.nn.functional.cross_entropy(clf(X), y).item(),
#            (clf(X).argmax(-1) == y).float().mean().item()
#    )

#    for _ in range(10000):
#        optimizer.zero_grad()
#        l = torch.nn.functional.cross_entropy(clf(X_subset), y_subset)
#        l.backward()
#        optimizer.step()


    results = []

    for pipeline, train_loader, val_loader, test_loader in zip(pipelines, train_loaders, val_loaders, test_loaders):
        print(pipeline)

        for data_type, loader in zip(['train', 'validation', 'test'], ([train_loader, val_loader, test_loader])):
            print(f'projecting {data_type}')

            X = []
            y = []
            with torch.no_grad():
                for index, (imgs, labels) in enumerate((loader)):
                    imgs = imgs.to(model.device)
                    X.append(model(imgs).cpu())
                    y.append(labels.cpu())

            X = torch.cat(X).to(model.device)
            y = torch.cat(y).to(model.device)

            print(f'evaluating {data_type}')

            preds = X
            
            r = {}
            r['dataset'] = config.data.name
            r['model'] = config.model.name
            r['transformation'] = config.transformations.type
            r['name'] = config.name
            r['checkpoint_id'] = ckpt_path.split('/')[-1].replace('.ckpt', '')
            r['checkpoint_path'] = ckpt_path
            r['train_percentage'] = TRAIN_PERCENTAGE
            r['train_test_split'] = data_type
            r['eval_pipeline_resize'] = pipeline.resize.size[0]
            r['eval_pipeline_pad'] = pipeline.pad.padding
            r['eval_pipeline_min_rotate'] = int(pipeline.rotate.degrees[0])
            r['eval_pipeline_max_rotate'] = int(pipeline.rotate.degrees[1])
            r['loss'] = torch.nn.functional.cross_entropy(preds, y).item()
            r['acc'] = (preds.argmax(-1) == y).float().mean().item()
            results.append(r)

    print('writing csv')
    filename = f"/data/out/evaluation_results2/{os.getenv('HOSTNAME', 'podname')}-{ckpt_index}-{time.time_ns()}.csv"
    pd.DataFrame.from_dict(results).to_csv(filename, index=False)

if __name__  ==  '__main__':

    with open('checkpoint_list') as f:
        ckpts = [i.strip() for i in f.readlines()]

    if len(sys.argv) >  2:
        try:
            l_index = int(sys.argv[1])
            r_index = int(sys.argv[2])
            ckpts = ckpts[l_index:r_index]
        except:
            print('failed to parse argv')

    for index, ckpt in enumerate(ckpts):
        print(ckpt)
        main(index, ckpt)
