#!/bin/env python
import sys
import yaml
import json
from string import Template
import subprocess
import argparse
import time
import collections

from dataclasses import dataclass

class JobFactory:
    def __init__(self, **kwargs):
        self.delete_only = kwargs['delete_only']
        self.dry_run = kwargs['dry_run']
        self.git_hash = kwargs['git_hash']
        self.time_delay = kwargs['delay']
        self.batch_size = kwargs['batch_size']

    def process_checkpoint_list(self, checkpoint_list):
        for l_index in range(0, len(checkpoint_list), self.batch_size):
            r_index = l_index + self.batch_size

            worker = Template(open('kubernetes/model_evalution.yaml', 'r').read())
            templated_worker = worker.substitute(
                l_index = l_index,
                r_index = r_index
            )

            self.create_kube_object(templated_worker)

            time.sleep(self.time_delay)

    def create_kube_object(self, spec):
        if self.dry_run:
            print(spec)
            return

        subprocess.run(['kubectl','delete', '-f', '-'], input=spec.encode())
        if not self.delete_only:
            subprocess.run(['kubectl','create', '-f', '-'], input=spec.encode())
        return 


def get_git_hash() -> str:
    git_status = subprocess.run(['git', 'ls-files', '-m'], text=True, capture_output=True).stdout.strip()
    if git_status != '':
        print('uncommitted changes')
        response = ''
        while response not in ['y', 'n']:
            response = input('should we proceed? (y/n)').strip().lower()

        if response.strip().lower() != 'y':
            print('exiting')
            sys.exit(1)
        print('proceeding')

    git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True).stdout.strip()
    return git_hash

def get_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', help='number of checkpoints to run in one pod', type=int, default=4)
    parser.add_argument('-d', '--delete-only', help='do not create any jobs, only delete jobs specified by checkpoint_list', action='store_true')  
    parser.add_argument('-n', '--dry-run', help='dry run (no actions committed on kube cluster)', action='store_true') 
    parser.add_argument('-t', '--delay', help='time delay', default=0, type=int)  
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cli_args = get_cli_args()
    git_hash = get_git_hash()

    with open('checkpoint_list') as f:
        checkpoint_list = [i.strip() for i in f.readlines()]

    job_factory = JobFactory(**vars(cli_args), git_hash = git_hash)
    job_factory.process_checkpoint_list(checkpoint_list)
