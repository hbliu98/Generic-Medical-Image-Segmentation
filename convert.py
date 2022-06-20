import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from typing import OrderedDict
from collections import OrderedDict


def brats19(in_dir, out_dir):
    """
    convert brats 2019 training data to required format
    """
    def _convert_labels(in_file, out_file):
        """
        convert brats labels from sparse {0, 1, 2, 4} to continuous {0, 1, 2, 3}
        """
        img = sitk.ReadImage(in_file)
        img_arr = sitk.GetArrayFromImage(img)

        seg_new = np.zeros_like(img_arr)
        seg_new[img_arr == 4] = 3
        seg_new[img_arr == 2] = 1
        seg_new[img_arr == 1] = 2
        img_corr = sitk.GetImageFromArray(seg_new)
        img_corr.CopyInformation(img)
        sitk.WriteImage(img_corr, out_file)
    
    if not os.path.exists(os.path.join(out_dir, 'imagesTr')):
        os.mkdir(os.path.join(out_dir, 'imagesTr'))
    if not os.path.exists(os.path.join(out_dir, 'labelsTr')):
        os.mkdir(os.path.join(out_dir, 'labelsTr'))
    
    all_names = []
    modalities = ['t1', 't1ce', 't2', 'flair']
    for tpe in ['HGG', 'LGG']:
        try:
            names = os.listdir(os.path.join(in_dir, tpe))
        except:
            names = []
        for name in names:
            if name == '.DS_Store':
                continue
            all_names.append(f'{tpe}__{name}')
            for i, modality in enumerate(modalities):
                shutil.copy(
                    os.path.join(in_dir, tpe, name, f'{name}_{modality}.nii.gz'),
                    os.path.join(out_dir, 'imagesTr', f'{tpe}__{name}_000{i}.nii.gz')
                )
            _convert_labels(
                os.path.join(in_dir, tpe, name, f'{name}_seg.nii.gz'),
                os.path.join(out_dir, 'labelsTr', f'{tpe}__{name}.nii.gz')
            )
    all_names.sort()
    # dataset.json
    json_dict = OrderedDict()
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['modality'] = {
        "0": "t1",
        "1": "t1ce",
        "2": "t2",
        "3": "flair"
    }
    json_dict['numTest'] = 0
    json_dict['numTraining'] = len(all_names)
    json_dict['training'] = [{'image': "imagesTr/%s.nii.gz" % i, "label": "labelsTr/%s.nii.gz" % i} for i in all_names]

    with open(os.path.join(out_dir, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    in_dir = 'brats19_toydata'
    out_dir = 'DATA/raw/brats19'
    brats19(in_dir, out_dir)
