# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import imageio
from configs.config2d import config
from networks.convcrf2d import ConvCRF2d
from utils.utils import augment_label
from utils.utils import plot_results
from utils.utils import process_img_unary

image_file = 'data/2007_000346_0img.png'
label_file = 'data/2007_000346_5labels.png'
image_show = imageio.imread(image_file)
label_show = imageio.imread(label_file)

model = ConvCRF2d(config, kernel_size=7).to(config.device)
unary_show = augment_label(label_show, num_classes=config.num_classes, scale=8, keep_prop=0.8)
image, unary = process_img_unary(image_show, unary_show)
image = torch.from_numpy(image).float().to(config.device)
unary = torch.from_numpy(unary).float().to(config.device)
output = model(image, unary)
output_show = output.data.cpu().numpy()
plot_results(image_show, unary_show, output_show, label_show)