# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class Config:
    gpus = [0,]
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

    num_classes = 34
    num_iters = 3
    momentum = 0.5
    downsample_rate = 4
    modes = ['pos', 'com']
    channels = [2, 5]

config = Config()