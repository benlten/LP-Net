import os
import yaml
import sys
import tarfile
from config import Config
import tqdm
import requests
from io import BytesIO
import time

def track_progress(tarobj):
    total_length = len(tarobj.getmembers())
    one_percent = total_length // 100
    for i, member in enumerate(tqdm.tqdm(tarobj.getmembers())):
        yield member

def timer(f):
    def timed(*args):
        t0 = time.time_ns()
        v = f(*args)
        print(f.__name__, (time.time_ns() - t0) / 1_000_000_000)
        return v
    return timed


@timer
def extract_nfs(dataset, location):
    with tarfile.open(f'/newawsmount/{dataset}.tar.gz') as f:
        f.extractall(location, members=track_progress(f))

@timer
def extract_aws(dataset, location):
    res = requests.get(f'http://s3-west.nrp-nautilus.io/shoob_imagenet/{dataset}.tar.gz', stream=True)
    download = BytesIO()

    block_size = 4096
    file_size = int(res.headers.get('Content-Length', None))
    for i, chunk in enumerate(tqdm.tqdm(res.iter_content(chunk_size=block_size), total=file_size // block_size)):
        download.write(chunk)

    download.seek(0)

    with tarfile.open(fileobj=download, mode='r|*') as f:
        f.extractall(location)

@timer
def extract_aws2(dataset, location):
    while True:
        try:
            res = requests.get(f'http://s3-west.nrp-nautilus.io/shoob_imagenet/{dataset}.tar.gz', stream=True)
            with tarfile.open(fileobj=res.raw, mode='r|*') as f:
                f.extractall(location)
            return
        except:
            print('Retrying...')
        


def download_github_folder(api_url, location, relative_path=""):
    # Make the GET request to fetch folder contents
    response = requests.get(api_url)

    if response.status_code == 200:
        # Create the extraction location if it doesn't exist
        os.makedirs(os.path.join(location,relative_path), exist_ok=True)

        # Parse the response JSON
        contents = response.json()

        # Iterate through the contents
        for item in contents:
            item_name = item["name"]
            item_type = item["type"]
            item_path = os.path.join(relative_path, item_name)

            if item_type == "file":
                file_url = item["download_url"]
                file_name = os.path.join(location, item_path)
                file_response = requests.get(file_url)
                with open(file_name, "wb") as file:
                    file.write(file_response.content)
                print(f"Downloaded: {file_name}")
            elif item_type == "dir":
                # Recursively download contents of subdirectories
                subdir_api_url = item["url"]
                download_github_folder(subdir_api_url, location, item_path)
    else:
        print(f"Failed to fetch folder contents. Status code: {response.status_code}")

@timer
def extract_github(dataset, location):
    # Hardcoded GitHub API URL for the folder
    api_url = "https://api.github.com/repos/rgeirhos/texture-vs-shape/contents/stimuli/style-transfer-preprocessed-512"

    print(f'starting to extract {dataset}', flush=True)
    download_github_folder(api_url, location , relative_path='imagenet/1000_identities/test')
    print(f'finished extracting {dataset}', flush=True)




if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = yaml.load(open(config_file, 'r'), Loader=yaml.SafeLoader)

    configs_to_run = [Config(**c) for c in config['runs']]
    if len(sys.argv) >  1:
        try:
            index = int(sys.argv[1])
            configs_to_run = [configs_to_run[index]]
        except:
            print('failed to parse argv')
    for run in (configs_to_run):
        dataset = run.data.name
        location = run.data.dataset
        print(f'starting to extract {dataset}', flush=True)
        extract_aws2(dataset, location)
#        extract_github(dataset, location)
        # extract_aws(dataset, location)
        # extract_nfs(dataset, location)
        print(f'finished extracting {dataset}', flush=True)
