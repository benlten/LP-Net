import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage.transform import rotate, rescale, resize
import pickle

# from retina_transform import foveat_img
import sys
sys.path.append('..')
from transformations import SalienceSampling, LogPolar, NRandomCrop, Compose

from torch.profiler import profile, record_function, ProfilerActivity

class CustomPickleDataset(Dataset):
    def __init__(self, pickle_path, crop_size, max_rotation, lp, lp_out_shape, augmentation, points, inversion, count = 1):
        super().__init__()

        with open(pickle_path, 'rb') as f:
            self.samples = pickle.load(f)

        self.count = count
        trans = []

        if augmentation == 'salience':
            trans.append(SalienceSampling(points, crop_size))
        elif augmentation == 'random': 
            trans.append(NRandomCrop(points, crop_size))
        else:
            trans.append(transforms.Resize(crop_size))

        if inversion:
            trans.append(transforms.RandomRotation((180,180)))
        else:
            trans.append(transforms.RandomRotation(max_rotation))

        if lp:
            trans.append(LogPolar(output_shape = lp_out_shape))

        self.augment = Compose(trans)

    def __getitem__(self, index):
        path, image, salience_map, target = self.samples[index]
        #  image = image.cuda()

        if not self.transform:
            return path, image, SalienceSampling.getSalienceMap(self.salience_path, path), target

        image = torch.stack([self.augment(image, path) for _ in range(count)])

        return image, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    dataset = FacesDataset(
            pickle_path = '/awsmount/16_ids.pkl',
            crop_size = 180, 
            max_rotation = 15, 
            lp = True, 
            lp_out_shape = (190, 165), 
            augmentation = 'salience', 
            points = 8,
            inversion = False)

    import tqdm
    (sum([1 for i in tqdm.tqdm(dataset)]))
