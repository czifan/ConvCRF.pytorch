# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import ceil, sqrt

class ConvCRF2d(nn.Module):
    def __init__(self, config, image_size, kernel_size):
        super().__init__()
        self.image_size = image_size
        self.num_classes = config.num_classes
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1, "'kernel_size' should be odd"
        self.span = self.kernel_size // 2
        self.blur = config.blur
        self.device = config.device
        self.col_bias = config.col_bias
        self.num_iters = config.num_iters
        self.momentum = config.momentum
        # self.comp = nn.Conv2d(self.num_classes, self.num_classes,
        #                       kernel_size=1, stride=1, padding=0, bias=False)
        # self.comp.weight.data.fill_(0.1 * sqrt(2.0 / self.num_classes))

        # pos_sdims = 3.
        # col_sdims = 2.
        # pos_compat = 3.
        # col_compat = 10.
        self._register_parameter('mesh', self._create_mesh())
        self._register_parameter('pos_sdims', torch.Tensor([1. / config.pos_sdims])) # 0.33
        self._register_parameter('col_sdims', torch.Tensor([1. / config.col_sdims])) # 0.5
        self._register_parameter('pos_compat', torch.Tensor([config.pos_compat]))
        self._register_parameter('col_compat', torch.Tensor([config.col_compat]))
        self._register_parameter('col_schan', torch.Tensor([1. / config.col_schan]))
        print(self.pos_sdims)
        print(self.col_sdims)
        print(self.pos_compat)
        print(self.col_compat)
        print(self.col_schan)

    def _add_pairwise_energies(self, img):
        B, C, H, W = img.shape

        pos_feats = self._create_position_feats(B)
        col_feats = self._create_colour_feats(img)

        self._norm_list = []
        self._gaus_list = []

        for feats, compat in zip([pos_feats, col_feats],
                                 [self.pos_compat, self.col_compat]):
            # gaussian : (B, 1, K, K, H_s, W_s)
            gaussian = self._create_convolutional_filters(feats)
            mynorm = self._compute_norm(gaussian)
            gaussian = compat * gaussian
            self._norm_list.append(mynorm)
            self._gaus_list.append(gaussian)

    def _compute_norm(self, gaussian):
        input = torch.ones([1, 1, *self.image_size]).to(self.device)
        # norm: (1, 1, H_l, W_l)
        norm = self._compute_gaussian(input, gaussian)
        return 1. / torch.sqrt(norm + 1e-20)

    def _compute_input(self, input):
        if self.blur > 1:
            off_0 = (self.blur - self.image_size[0] % self.blur) % self.blur
            off_1 = (self.blur - self.image_size[1] % self.blur) % self.blur
            pad_0 = int(ceil(off_0 / 2))
            pad_1 = int(ceil(off_1 / 2))
            input = F.avg_pool2d(input, kernel_size=self.blur,
                                 padding=(pad_0, pad_1), count_include_pad=False)
            down_image_size = [ceil(self.image_size[0] / self.blur),
                               ceil(self.image_size[1] / self.blur)]
        else:
            pad_0 = 0
            pad_1 = 0
            down_image_size = self.image_size

        return input, down_image_size, pad_0, pad_1

    def _compute_gaussian(self, input, gaussian, norm=None):
        # gaussian : (B, 1, K, K, H_s, W_s)
        # input: (B, num_classes, H_l, W_l)
        shape = input.shape

        if norm is not None:
            input = input * norm

        # input: (B, num_classes, H_s, W_s)
        input, down_image_size, pad_0, pad_1 = self._compute_input(input)
        B, C, H, W = input.shape
        assert (down_image_size[0] == H) and (down_image_size[1] == W)
        input_unfold = F.unfold(input, self.kernel_size, 1, self.span)
        input_unfold = input_unfold.view(B, C, self.kernel_size, self.kernel_size, H, W)
        # input_col: (B, num_classes, K, K, H_s, W_s)
        input_col = input_unfold

        product = gaussian * input_col
        product = product.view([B, C, self.kernel_size**2, H, W])
        # message : (B, num_classes, H_s, W_s)
        message = product.sum(dim=2)

        if self.blur > 1:
            message = message.view(B, C, H, W)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                message = F.upsample(message, scale_factor=self.blur, mode='bilinear')
            message = message[:, :,
                              pad_0:pad_0+self.image_size[0],
                              pad_1:pad_1+self.image_size[1]].contiguous()
            message = message.view(shape)

        if norm is not None:
            message = message * norm

        # message: (B, num_classes, H_l, W_l)
        return message

    def _create_convolutional_filters(self, feats):
        # [input]  feats    : (B, C, H_l, W_l)
        # [output] feats    : (B, C, H_s, W_s)
        # [input vs. output]: H_s = H_l // self.blur, W_s = W_l // self.blur
        feats, down_image_size, pad_0, pad_1 = self._compute_input(feats)

        # gaussian : (B, K, K, H_s, W_s)
        gaussian = feats.data.new(feats.shape[0],
                                  self.kernel_size, self.kernel_size,
                                  *down_image_size).fill_(0).to(self.device)

        for dx in range(-self.span, self.span+1):
            for dy in range(-self.span, self.span+1):
                dx1, dx2 = self._get_ind(dx)
                dy1, dy2 = self._get_ind(dy)
                feat_t1 = feats[:, :,
                                dx1:self._negative(dx2),
                                dy1:self._negative(dy2)]
                feat_t2 = feats[:, :,
                                dx2:self._negative(dx1),
                                dy2:self._negative(dy1)]
                diff_sq = (feat_t1 - feat_t2) ** 2
                exp_diff_sq = torch.exp(torch.sum(-0.5 * diff_sq, dim=1))
                gaussian[:, dx+self.span, dy+self.span,
                         dx2:self._negative(dx1), dy2:self._negative(dy1)] = exp_diff_sq

        # gaussian : (B, 1, K, K, H_s, W_s)
        gaussian = gaussian.view(feats.shape[0], 1,
                                 self.kernel_size, self.kernel_size,
                                 *down_image_size)

        return gaussian

    def _compute(self, input):
        assert (len(self._gaus_list) == len(self._norm_list))
        prediction = 0
        for gaus, norm in zip(self._gaus_list, self._norm_list):
            prediction += self._compute_gaussian(input, gaus, norm)
        return prediction

    def _create_mesh(self):
        hcord_range = [range(s) for s in self.image_size]
        mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)
        mesh = torch.from_numpy(mesh).to(self.device)
        return mesh

    def _create_colour_feats(self, img):
        norm_img = img * self.col_schan
        if self.col_bias:
            norm_mesh = self._create_position_feats(img.shape[0])
            feats = torch.cat([norm_mesh, norm_img], dim=1)
        else:
            feats = norm_img
        return feats

    def _create_position_feats(self, B):
        return torch.stack(B * [self.mesh * self.col_sdims])

    def _register_parameter(self, name, tensor):
        self.register_parameter(name, Parameter(tensor))

    def _get_ind(self, dz):
        if dz == 0:
            return 0, 0
        if dz < 0:
            return 0, -dz
        if dz > 0:
            return dz, 0

    def _negative(self, dz):
        if dz == 0:
            return None
        else:
            return -dz

    def forward(self, img, unary):
        self._add_pairwise_energies(img)
        prediction = F.log_softmax(unary, dim=1, _stacklevel=5)

        for i in range(self.num_iters):
            message = self._compute(prediction)
            # comp = self.comp(message)
            # message = message + comp
            prediction = (1. - self.momentum) * prediction + self.momentum * message

        return prediction


