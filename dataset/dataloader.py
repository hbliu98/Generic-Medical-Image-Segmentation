import os
import pickle
import numpy as np
from core.config import config
from collections import OrderedDict
from utils.utils import get_patch_size
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1/3
        # larger patch size is required for proper data augmentation
        self.patch_size = get_patch_size(patch_size, (-np.pi, np.pi), (0, 0), (0, 0), (0.7, 1.4))
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form slice
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])['data']
            # slice with foreground class presented should be selected ?
            if i < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                force_fg = False
            else:
                force_fg = True
            if force_fg:
                # select slice containing foreground class
                locs = self._data[name]['locs']
                cls = np.random.choice(list(locs.keys()))
                indices = locs[cls][:, 0]   # all axial indices
                sel_idx = np.random.choice(np.unique(indices))
                data = data[:, sel_idx]
                # pad slice centered at selected location
                # the idea is simple: if selected location is biased towards left, then left side will be padded more pixels. note that we cannot crop slice, so the minimum pad length is zero
                loc = locs[cls][indices == sel_idx]
                loc = loc[np.random.choice(len(loc))][1:]
                shape = np.array(data.shape[1:])
                center = shape // 2
                bias = loc - center
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2 - bias
                pad_right = pad_length - pad_length // 2 + bias
                pad_left = np.clip(pad_left, 0, pad_length)
                pad_right = np.clip(pad_right, 0, pad_length)
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            else:
                # randomly select slice
                sel_idx = np.random.choice(data.shape[1])
                data = data[:, sel_idx]
                shape = np.array(data.shape[1:])
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2
                pad_right = pad_length - pad_length // 2
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1/3
        # larger patch size is required for proper data augmentation
        self.patch_size = get_patch_size(patch_size, (-np.pi/6, np.pi/6), (-np.pi/6, np.pi/6), (-np.pi/6, np.pi/6), (0.7, 1.4))
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form patch
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])['data']
            # patch with foreground class presented should be selected ?
            if i < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                force_fg = False
            else:
                force_fg = True
            if force_fg:
                # select patch containing foreground class
                locs = self._data[name]['locs']
                cls = np.random.choice(list(locs.keys()))
                locs = locs[cls]
                loc = locs[np.random.choice(len(locs))]
            else:
                # randomly select patch, centered at (sel_z, sel_x, sel_y)
                sel_z = np.random.choice(data.shape[1])
                sel_x = np.random.choice(data.shape[2])
                sel_y = np.random.choice(data.shape[3])
                loc = np.array((sel_z, sel_x, sel_y))
            # crop
            shape = np.array(data.shape[1:])
            left = np.clip(loc - self.patch_size // 2, a_min=0, a_max=None)
            right = np.clip(loc + (self.patch_size - self.patch_size // 2), a_min=None, a_max=shape)
            data = data[:, left[0]:right[0], left[1]:right[1], left[2]:right[2]]
            # pad
            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}


def get_trainloader(fold):
    # list data path and properties
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    trains = splits['train']
    dataset = OrderedDict()
    for name in trains:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name+'.npz')
        with open(os.path.join(config.DATASET.ROOT, name+'.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)
    # 2d or 3d ?
    if config.MODEL.NUM_DIMS == 2:
        assert len(config.TRAIN.PATCH_SIZE) == 2, 'must be 2 dimensional patch size'
        return DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)
    elif config.MODEL.NUM_DIMS == 3:
        assert len(config.TRAIN.PATCH_SIZE) == 3, 'must be 3 dimensional patch size'
        return DataLoader3D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)
