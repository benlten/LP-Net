import lmdb
import os
from PIL import Image
import numpy as np
import tqdm

# Set your dataset directory path
dataset_dir = '/awsmount/faces/128_identities/train'

# Set the path where you want to store the lmdb dataset
lmdb_dir = '/awsmount/lmdb/faces/'
os.makedirs(lmdb_dir, exist_ok = True)

# Create the lmdb environment
env = lmdb.open(lmdb_dir, map_size=int(1e11))

# Loop through all JPEG files in your dataset directory
for directory in tqdm.tqdm(os.listdir(dataset_dir)):
    for filename in os.listdir(os.path.join(dataset_dir, directory)):
        # Open the lmdb transaction
        with env.begin(write=True) as txn:
            img = Image.open(os.path.join(dataset_dir, directory, filename))
            img_arr = np.array(img)
            img_bytes = img_arr.tobytes()
            
            # Add the image to the lmdb database
            key = str(os.path.join(dataset_dir, directory, filename))
            txn.put(key.encode('ascii'), img_bytes)
            txn.put((key+'shape').encode('ascii'), np.array(img_arr.shape).tobytes())

# Close the lmdb environment
env.close()
