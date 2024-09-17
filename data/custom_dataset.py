import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

import os

from torch.profiler import profile, record_function, ProfilerActivity
from utils import MemoryMonitor

class TensorBackedImmutableStringArray:
    def __init__(self, strings, encoding = 'utf-8'):
        encoded = [torch.ByteTensor(torch.ByteStorage.from_buffer(s.encode(encoding))) for s in strings]
        self.cumlen = torch.cat((torch.zeros(1, dtype = torch.int64), torch.as_tensor(list(map(len, encoded)), dtype = torch.int64).cumsum(dim = 0)))
        self.data = torch.cat(encoded)
        self.encoding = encoding

    def __getitem__(self, i):
        return bytes(self.data[self.cumlen[i] : self.cumlen[i + 1]]).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen) - 1

    def __list__(self):
        return [self[i] for i in range(len(self))]

class CustomDataset(datasets.ImageFolder):
    def __init__(self, data_path, *, augment=lambda x: x, find_classes_data_path = None, repetitions=1):
        self.data_path = data_path
        self.repetitions = repetitions

        self.augment = augment

        # self.tensorize = transforms.ToTensor()

        self.find_classes_data_path = data_path if find_classes_data_path is None else find_classes_data_path

        super().__init__(data_path)
        self.samples = self.samples
        self.imgs = ([i for (i,_) in self.samples])


        self.samples = TensorBackedImmutableStringArray([i for (i,_) in self.samples])
        self.imgs = self.samples
        self.targets = torch.tensor(self.targets )

    def __getitem__(self, index):
        index = index % len(self.imgs)
        path = self.imgs[index]
        target = self.targets[index]

        # image = self.tensorize(image) # tensorize
        image = self.augment(self.loader(path))

        # print('data', index, MemoryMonitor().str(), flush=True)

        return image, target
        
    def __len__(self):
        return len(self.imgs) * self.repetitions

    def find_classes(self, directory):
        path = os.path.join(self.find_classes_data_path, directory[directory.rindex('/')+1:])
        return super().find_classes(path)

    def make_dataset(self, directory, class_to_idx, extensions, is_valid_file):
        def has_file_allowed_extension(filename, extensions):
            return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        if target_class not in available_classes:
                            available_classes.add(target_class)

        return instances

if __name__ == '__main__':
    dataset = FacesDataset(
            data_path = '/awsmount/faces/16_identities',
            salience_path = '/awsmount/salience_maps/16_identities',
            crop_size = 180,
            max_rotation = 15,
            lp = True,
            lp_out_shape = (190, 165),
            augmentation = 'random',
            points = 8,
            inversion = False, # transform=False
            )

    import pdb; pdb.set_trace()
    import tqdm
    print(sum([1 for i in tqdm.tqdm(dataset)]))
