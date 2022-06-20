import torch
import numpy as np
import torch.nn as nn
import torchio as tio
from collections import OrderedDict
from medpy.metric.binary import dc

from utils.utils import AverageMeter, determine_device


def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    losses = AverageMeter()
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()
    num_iter = config.TRAIN.NUM_BATCHES
    for i in range(num_iter):
        data_dict = next(train_generator)
        data = data_dict['data']
        label = data_dict['label']
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            data = data.cuda(devices[0])
            label = [l.cuda(devices[0]) for l in label]
        else:
            device = determine_device()
            data = data.to(device)
            label = [l.to(device) for l in label]
        # run training
        with torch.cuda.amp.autocast():
            out = model(data)
            loss = criterion(out, label)
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, i, num_iter,
                    loss = losses,
                )
            logger.info(msg)


def inference(model, dataset, logger, config):
    model.eval()
    
    num_classes = config.DATASET.NUM_CLASSES
    perfs = [AverageMeter() for _ in range(num_classes)]
    nonline = nn.Softmax(dim=1)
    scores = {}
    for case in dataset:
        dims = config.MODEL.NUM_DIMS
        patch_size = config.INFERENCE.PATCH_SIZE
        patch_overlap = config.INFERENCE.PATCH_OVERLAP
        # pad data to match patch size
        target_shape = [case['data'][tio.DATA].shape[1]] + patch_size
        transform = tio.CropOrPad(target_shape)
        case = transform(case)
        # torchio does not support 2d slice natively, it can only treat it as pseudo 3d patch
        if dims == 2:
            patch_size = [1] + patch_size
            patch_overlap = [0] + patch_overlap
        sampler = tio.inference.GridSampler(case, patch_size, patch_overlap)
        loader = torch.utils.data.DataLoader(sampler, config.INFERENCE.BATCH_SIZE)
        aggregator = tio.inference.GridAggregator(sampler, 'average')

        with torch.no_grad():
            for data_dict in loader:
                data = data_dict['data'][tio.DATA]
                label = data_dict['label'][tio.DATA]
                if dims == 2:
                    data = data.squeeze(2)
                    label = label.squeeze(2)
                if config.TRAIN.PARALLEL:
                    devices = config.TRAIN.DEVICES
                    data = data.cuda(devices[0])
                    label = label.cuda(devices[0])
                else:
                    device = determine_device()
                    data = data.to(device)
                    label = label.to(device)
                with torch.cuda.amp.autocast():
                    out = model(data)[0]
                    out = nonline(out)
                locations = data_dict[tio.LOCATION]
                # I love and hate torchio ...
                if dims == 2:
                    out = out.unsqueeze(2)
                aggregator.add_batch(out, locations)
            # form final prediction
            pred = aggregator.get_output_tensor()
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = case['label'][tio.DATA][0].numpy()
            name = case['name']
            # quantitative analysis
            # only dice score is computed by default, you can also add hd95, assd and sensitivity et al
            scores[name] = {}
            for c in np.unique(label):
                scores[name][int(c)] = dc(pred==c, label==c)
                perfs[int(c)].update(scores[name][c])
    logger.info('------------ dice scores ------------')
    logger.info(scores)
    for c in range(num_classes):
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in range(1, num_classes)])
    return perf
