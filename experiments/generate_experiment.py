#!/bin/env python
import sys
import yaml
from string import Template
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', nargs='*', default=['faces', 'dogs', 'imagenet100'])
parser.add_argument('-m', '--models', nargs='*', default=['simclr','byol', 'barlow_twins', 'siamese_net'])
parser.add_argument('-t', '--transformations', nargs='*', default=['simclr','custom', 'hemisphere'])
parser.add_argument('--lp', nargs='*', default=[False, True])
args = parser.parse_args()
print(args)

template_file = 'template.yaml'

jobs = []

for dataset in args.datasets:
    for model in args.models:
        for transformation in args.transformations:
            for lp in args.lp:
                template = yaml.load(open(template_file, 'r'), Loader=yaml.SafeLoader)
                template['transformations']['type'] = transformation
                template['transformations']['log_polar']['active'] = lp
                template['model']['name'] = model
                template['data']['name'] = dataset

                if dataset == 'imagenet100':
                    template['data']['classes'] = [100]
                    template['data']['find_classes_data_path'] = 100
                else:
                    template['data']['classes'] = [128]
                    template['data']['find_classes_data_path'] = 128

                template['name'] = f'{model}_{transformation}_{"default" if not lp else "lp"}'

                jobs.append(template)
yaml.dump({'runs': jobs}, open('output.yaml', 'w'), indent=2, )
print('Created output file with', len(jobs), 'jobs to execute')
