import torch
import argparse
import torch.nn as nn

from models import *
from core.config import config
from core.function import inference
from dataset.dataset import get_testset
from utils.utils import determine_device, update_config, create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='configuration file', required=True, type=str)
    parser.add_argument('--weights', help='path for pretrained weights', required=True, type=str)
    parser.add_argument('--root', help='path for testing data', required=True, type=str)
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
    # load pretrained weights
    model.load_state_dict(torch.load(args.weights))
    # validation dataset
    validset = get_testset(args.root)

    logger = create_logger('log', 'test.log')
    inference(model, validset, logger, config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
