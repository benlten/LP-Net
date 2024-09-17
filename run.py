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
        self.force = kwargs['force']
        self.dry_run = kwargs['dry_run']
        self.git_hash = kwargs['git_hash']
        self.time_delay = kwargs['delay']
        self.auto_wait = kwargs['auto_wait']

    def process_config(self, config):
        for jobidx,run in enumerate(config['runs']):
            all_jobs = self.create_job(jobidx, run)
            if self.auto_wait and all_jobs is not None:
                self.wait_for_jobs(all_jobs)
            else:
                time.sleep(self.time_delay)


    def create_job(self, jobidx, run):
        if not self.force:
            print(f'deciding job {jobidx}')
            # print(yaml.dump(run, default_flow_style=False, allow_unicode=True))
            response = input()
            if response != 'y':
                print(f'skipping job {jobidx}')
                return None

        service = Template(open('kubernetes/multi_node/service.yaml', 'r').read())
        worker = Template(open('kubernetes/multi_node/worker.yaml', 'r').read())
          
        trunc_hash = self.git_hash[:8]

        templated_service = service.substitute(
                jobidx = jobidx,
                trunc_hash=trunc_hash
        )

        self.create_kube_object(templated_service)

        total_nodes = run['compute'].get('num_nodes', 2)
        all_jobs = []
        for nodeidx in range(total_nodes):
            templated_worker = (worker.substitute(
                jobidx=jobidx, 
                total=total_nodes,
                nodeidx=nodeidx,
                git_hash=self.git_hash,
                trunc_hash=trunc_hash,
                endpoint=f'mn-job{jobidx}-commit{trunc_hash}-router' if nodeidx != 0 else 'localhost'
            ))

            job_name = self.create_kube_object(templated_worker)
            all_jobs.append(job_name)
        return all_jobs


    def create_kube_object(self, spec):
        job_name = yaml.load(spec, Loader=yaml.Loader)['metadata']['name']
        if self.dry_run:
            print(spec)
            return job_name

        subprocess.run(['kubectl','delete', '-f', '-'], input=spec.encode())
        if not self.delete_only:
            subprocess.run(['kubectl','create', '-f', '-'], input=spec.encode())
        return job_name

    def wait_for_jobs(self, jobs):
        if self.delete_only or self.dry_run:
            return
        all_running = False
        i = 0
        while not all_running:
            time.sleep(self.time_delay)
            pods = [
                pod 
                for pod in 
                json.loads(
                    subprocess.run(['kubectl', 'get', 'pods', '-o', 'json'], text=True, capture_output=True).stdout.strip()
                )['items'] 
                if (
                    pod['metadata']
                        .get('labels', {})
                        .get('job-name', '') 
                    in jobs
                )
            ]
            run_counter = collections.Counter([pod['status']['phase'] for pod in pods])
            all_running = run_counter['Running'] == len(jobs)
            print(f'{"_"*40}iteration {i}: {run_counter}{"_"*40}', end='\r')
            i += 1
        print('\n')


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
    parser.add_argument('config', nargs='?', help='config file location', default='config.yaml')
    parser.add_argument('-d', '--delete-only', help='do not create any jobs, only delete jobs specified by config.yaml', action='store_true')  
    parser.add_argument('-f', '--force', help='no prompt', action='store_true') 
    parser.add_argument('-n', '--dry-run', help='dry run (no actions committed on kube cluster)', action='store_true') 
    parser.add_argument('-t', '--delay', help='time delay', default=0, type=int)  
    parser.add_argument('--auto', help='wait type', action='store_true', dest='auto_wait')  
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cli_args = get_cli_args()
    git_hash = get_git_hash()
    config = yaml.load(open(cli_args.config, 'r'), Loader=yaml.SafeLoader)
    job_factory = JobFactory(**vars(cli_args), git_hash = git_hash)
    job_factory.process_config(config)
