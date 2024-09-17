import lmdb
import cv2
import numpy as np

# Set the path to your lmdb dataset
lmdb_dir = '/awsmount/lmdb/faces'

# Open the lmdb environment in readonly mode
env = lmdb.open(lmdb_dir, readonly=True)

# Open a cursor to iterate through the lmdb database
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        # Decode the key and value from bytes to string and numpy array respectively
        key_str = key.decode('ascii')
        img_array = np.frombuffer(value, dtype=np.uint8)
        print(img_array.shape)

# Close the lmdb environment
env.close()
