import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models import *
from core.config import config
from core.scheduler import PolyScheduler
from core.function import train, inference
from core.loss import DiceCELoss, MultiOutLoss
from dataset.dataset import get_validset
from dataset.dataloader import get_trainloader
from dataset.augmenter import get_train_generator
from utils.utils import determine_device, save_checkpoint, update_config, create_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='configuration file', required=True, type=str)
    parser.add_argument('--fold', help='which data fold to train on', required=True, type=int)
    args = parser.parse_args()
    return args

def main(args):
    update_config(config, args.cfg)

    net = unet.UNet  # use your network architecture here --> <file_name>.<class_name>
    if config.TRAIN.PARALLEL:   # only cuda is supported
        devices = config.TRAIN.DEVICES
        model = net(config)
        model = nn.DataParallel(model, devices).cuda(devices[0])
    else:   # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = net(config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.99, nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)
    # deep supervision weights, normalize sum to 1
    criterion = DiceCELoss()
    weights = np.array([1 / (2 ** i) for i in range(len(config.MODEL.EXTRA.ENC_CHANNELS)-2)])
    weights /= weights.sum()
    criterion = MultiOutLoss(loss_function=criterion, weights=weights)
    # training data generator
    trainloader = get_trainloader(args.fold)
    train_generator = get_train_generator(trainloader)
    # validation dataset
    validset = get_validset(args.fold)

    best_model = False
    best_perf = 0.0
    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))
        
        # train(model, train_generator, optimizer, criterion, logger, config, epoch)
        # scheduler.step()
        # running validation at every epoch is time consuming
        if epoch%config.VALIDATION_INTERVAL == 0:
            perf = inference(model, validset, logger, config)
        
        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, config.OUTPUT_DIR, filename='checkpoint.pth')


if __name__ == '__main__':
    args = parse_args()
    main(args)
