# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from configs.config3d import config
from networks.convcrf3d import ConvCRF3d

model = ConvCRF3d(config, kernel_size=7).to(config.device)
image = torch.Tensor(1, 1, 64, 64, 64).float().to(config.device)
unary = torch.Tensor(1, config.num_classes, 64, 64, 64).float().to(config.device)
spacing = torch.Tensor(1, 3).float().to(config.device)
output = model(image, unary, spacing)
print(output.shape)