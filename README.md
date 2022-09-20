# Generic Medical Image Segmentation Framework

> This is a generic medical image segmentation framework implemented by huabing liu (@hbliu98). The project structure is motivated by [human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch). The training pipeline is motivated by [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

This framework is implemented in many of our works. Please cite our work if you find it useful in your research:
```
@inproceedings{liu2022multimodal,
  title={Multimodal Brain Tumor Segmentation Using Contrastive Learning Based Feature Comparison with Monomodal Normal Brain Images},
  author={Liu, Huabing and Nie, Dong and Shen, Dinggang and Wang, Jinda and Tang, Zhenyu},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={118--127},
  year={2022},
  organization={Springer}
}
```

## Installation
The following environments/libraries are required:
- pytorch
- SimpleITK
- yacs
- scikit-learn
- medpy
- torchio

## Usage
1. Run convert.py. Convert data to required format, save into DATA/raw
2. Run preprocess.py. The preprocessed data are saved into DATA/preprocessed
3. Run train.py

You can use brats19_toydata for firing the whole framework, enjoy yourself 😊

## Folders
- DATA
  - raw: formatted raw data files, see [Data Organization](#data-organization)
  - preprocessed: preprocessed data files, i.e., crop and normalization
- dataset: data loading and data augmentation
- core
  - config.py: essential hyperparameters
  - function.py: training and testing codes
  - loss.py: codes for loss function
  - scheduler.py: learning rate scheduler
- configs: extra hyperparameters, stored in .yaml
- pretrained: pretrained network parameters
- experiments: outputs of experiment, e.g., checkpoint
- models: codes for segmentation networks
- log: log file
- tmp: for temporary files
- utils: useful codes
- preprocess.py, test.py and train.py: as the name suggests
- convert.py: convert dataset to required format

## Data Organization
- We follow the naming convention used in [nnUNet](https://github.com/MIC-DKFZ/nnUNet): <dataset_name>\_<case_identifier>\_<modality_identifier>.nii.gz
- imagesTr, labelsTr; imagesVal, labelsVal; imagesTs, labelsTs are directories to store training / validation / testing images and labels

## Extending
- For using a different network architecture, simply put network file in models folder and network configuration file in configs folder. Then just modify train.py accordingly
