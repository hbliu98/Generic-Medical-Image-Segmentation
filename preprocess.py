import os
import json
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from sklearn.model_selection import KFold
from scipy.ndimage import binary_fill_holes


def get_bbox(mask): # using SimpleITK, we get (z, x, y)
    coords = np.where(mask != 0)
    minz = np.min(coords[0])
    maxz = np.max(coords[0]) + 1
    minx = np.min(coords[1])
    maxx = np.max(coords[1]) + 1
    miny = np.min(coords[2])
    maxy = np.max(coords[2]) + 1
    return slice(minz, maxz), slice(minx, maxx), slice(miny, maxy)


def preprocess(raw_dir, preprocessed_dir):
    seed = 12345

    with open(os.path.join(raw_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    modality = dataset_json['modality']
    training = dataset_json['training']
    shapes, names = [], []
    for case in training:
        images = []
        for i in range(len(modality)):
            image = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7]+'_{:0>4d}'.format(i)+'.nii.gz'))
            image = sitk.GetArrayFromImage(image).astype('float')
            images.append(image)
        label = sitk.ReadImage(os.path.join(raw_dir, case['label']))
        label = sitk.GetArrayFromImage(label)
        image = np.stack(images)
        name = os.path.basename(case['label'][:-7])
        names.append(name)
        # crop to non_zero regions
        mask = np.zeros(image.shape[1:], dtype=bool)
        for i in range(len(modality)):
            mask = mask | (image[i] != 0)
        mask = binary_fill_holes(mask)
        bbox = get_bbox(mask)
        image = image[:, bbox[0], bbox[1], bbox[2]]
        label = label[bbox[0], bbox[1], bbox[2]]
        mask = mask[bbox[0], bbox[1], bbox[2]]
        shapes.append(label.shape)
        # intensity normalization within foreground
        for i in range(len(modality)):
            image[i][mask] = (image[i][mask] - image[i][mask].mean()) / (image[i][mask].std() + 1e-8)
            image[i][mask == 0] = 0 # all modalities share same background
        # pick enough locations for each class -> address imbalanced labels
        # follow nnUNet, 0.1 * all_locs <= num_locs <= all_locs
        classes = dataset_json['labels']
        class_locs = OrderedDict()
        approx_num = 10000
        rdst = np.random.RandomState(seed)
        for c in list(classes.keys())[1:]:
            all_locs = np.argwhere(label == int(c))
            num_locs = min(approx_num, len(all_locs))
            num_locs = max(num_locs, int(np.ceil(0.1 * len(all_locs))))
            sel = all_locs[rdst.choice(len(all_locs), num_locs, replace=False)]
            if len(sel) != 0:
                class_locs[c] = sel
        assert len(class_locs) != 0, f'whoops! {name} has no ROI...'
        # save preprocessed data and selected locations
        data = np.concatenate([image, label[np.newaxis]])
        # we do not use compressed .npz file since it requires a lot more cpu time to load
        # time is more precious than disk space, isn't it ?
        np.savez(os.path.join(preprocessed_dir, name+'.npz'), data=data.astype('float32'))
        with open(os.path.join(preprocessed_dir, name+'.pkl'), 'wb') as f:
            pickle.dump(class_locs, f)
    # statistics
    print('median data shape: {}'.format(np.median(shapes, 0)))
    print('maximum data shape: {}'.format(np.max(shapes, 0)))
    print('minimum data shape: {}'.format(np.min(shapes, 0)))
    # split data - 5 fold
    splits = []
    names = np.sort(names)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in kfold.split(names):
        trains = names[train_idx]
        vals = names[val_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = trains
        splits[-1]['val'] = vals
    with open(os.path.join(preprocessed_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


if __name__ == '__main__':
    raw_dir = 'DATA/raw/brats19'
    preprocessed_dir = 'DATA/preprocessed/brats19'
    preprocess(raw_dir, preprocessed_dir)
