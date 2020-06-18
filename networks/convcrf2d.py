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
    def __init__(self, config, kernel_size):
        super().__init__()

        # fixed parameters
        self.num_classes = config.num_classes
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1, "'kernel_size' should be odd"
        self.span = self.kernel_size // 2
        self.device = config.device
        self.downsample_rate = config.downsample_rate
        self.num_iters = config.num_iters
        self.momentum = config.momentum
        self.modes = config.modes
        self.channels = config.channels

        # learnable parameters
        self.thetas = [self.register_parameter('{}_theta'.format(mode), nn.Parameter(torch.ones(1, c, 1, 1).to(config.device) * 1.))
                            for c, mode in zip(self.channels, self.modes)]
        self.weights = [self.register_parameter('{}_weight'.format(mode), nn.Parameter(torch.Tensor([1.]).to(config.device))) for mode in self.modes]

    def _create_mesh(self, image_size):
        ''' creating 2d mesh according to the spacing of image
        :param image_size: np.array, (H, W)
        :return: 2d mesh: torch.Tensor, (2, H, W)
        '''
        hcord_range = [range(s) for s in image_size]
        mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'), dtype=np.float32)
        mesh = torch.from_numpy(mesh).to(self.device)
        return mesh

    def _generate_features(self, img, mode):
        ''' generating features
        :param img: torch.Tensor, (B, C_img, H, W)
        :param mode: str, gaussian kernel feature construction mode: pos (position) | col (colour) | com (combine)
        :return: features: torch.Tensor, (B, C_features, H, W)
        '''
        if mode == 'pos':
            return torch.stack(img.shape[0] * [self._create_mesh(img.shape[-2:])])
        elif mode == 'col':
            return img
        elif mode == 'com':
            return torch.cat([torch.stack(img.shape[0] * [self._create_mesh(img.shape[-2:])]), img], dim=1)
        else:
            print("'{}' is a no defined pattern.".format(mode))
            return None

    def _downsample(self, input):
        ''' Downsampling is performed to reduce the computation and expand the receptive field
        :param input: torch.Tensor, (B, C_input, H, W)
        :return: down_input: torch.Tensor, (B, C_input, H_s, W_s)
                 down_image_size: list, (H_s, W_s)
                 pad_0: float
                 pad_1: float
        '''
        image_size = input.shape[-2:]
        if self.downsample_rate > 1:
            off_0 = (self.downsample_rate - image_size[0] % self.downsample_rate) % self.downsample_rate
            off_1 = (self.downsample_rate - image_size[1] % self.downsample_rate) % self.downsample_rate
            pad_0 = int(ceil(off_0 / 2))
            pad_1 = int(ceil(off_1 / 2))
            down_input = F.avg_pool2d(input, kernel_size=self.downsample_rate,
                                 padding=(pad_0, pad_1), count_include_pad=False)
            down_image_size = [ceil(image_size[0] / self.downsample_rate),
                               ceil(image_size[1] / self.downsample_rate)]
        else:
            pad_0 = 0
            pad_1 = 0
            down_image_size = image_size
            down_input = input

        return down_input, down_image_size, pad_0, pad_1

    def _upsample(self, input, image_size, pad_0, pad_1):
        ''' Upsampling to restore original size
        :param input: torch.Tensor, (D, C_input, H_s, W_s)
        :param image_size: list, (H, S)
        :param pad_0: float
        :param pad_1: float
        :return: input: torch.Tensor, (D, C_input, H, W)
        '''
        if self.downsample_rate > 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                input = F.upsample(input, scale_factor=self.downsample_rate, mode='bilinear')
            input = input[:, :,
                              pad_0:pad_0+image_size[0],
                              pad_1:pad_1+image_size[1]].contiguous()
        return input

    def _convolution(self, input, gaussian_kernel):
        ''' Computing convolution operation
        :param input: torch.Tensor, (B, C_input, D_s, H_s, W_s)
        :param gaussian_kernel: torch.Tensor, (B, 1, K_d, K_h, K_w, D_s, H_s, W_s)
        :param compa: function or None, compatibility transformation
        :return: message, torch.Tensor, (B, C_cls, D_s, H_s, W_s)
        '''
        B, C, H, W = input.shape # (B, C_input, H_s, W_s)
        # input_col=input_unfold: (B, C_cls, K_h, K_w, H_s, W_s)
        input_unfold = F.unfold(input, self.kernel_size, 1, self.span)
        input_unfold = input_unfold.view(B, C, self.kernel_size, self.kernel_size, H, W)
        input_col = input_unfold
        # product: (B, 1, K_h, K_w, H_s, W_s) * (B, C_cls, K_h, K_w, H_s, W_s) --> (B, C_cls, K_h, K_w, H_s, W_s)
        product = gaussian_kernel * input_col
        # product: (B, C_cls, -1, H_s, W_s)
        product = product.view([B, C, self.kernel_size**2, H, W])
        # message: (B, C_cls, H_s, W_s)
        message = product.sum(dim=2)

        return message

    def _generate_convolutional_filters(self, feats, down_image_size, theta):
        ''' Generating convolutional filter according features
        :param feats: torch.Tensor, (B, C_features, H_s, W_s)
        :param down_image_size: list, (H_s, W_s)
        :param theta: torch.nn.Parameter, scaling parameters
        :return: gaussian: torch.Tensor, (B, 1, K_h, K_w, H_s, W_s)
        '''
        def _get_ind(dz):
            if dz == 0:
                return 0, 0
            if dz < 0:
                return 0, -dz
            if dz > 0:
                return dz, 0

        def _negative(dz):
            if dz == 0:
                return None
            else:
                return -dz

        # feats : (B, C_features, H_s, W_s)
        # gaussian : (B, K_h, K_w, H_s, W_s)
        gaussian = feats.data.new(feats.shape[0],
                                  self.kernel_size, self.kernel_size,
                                  *down_image_size).fill_(0).to(self.device)

        for dx in range(-self.span, self.span+1):
            for dy in range(-self.span, self.span+1):
                dx1, dx2 = _get_ind(dx)
                dy1, dy2 = _get_ind(dy)
                feat_t1 = feats[:, :,
                                dx1:_negative(dx2),
                                dy1:_negative(dy2)]
                feat_t2 = feats[:, :,
                                dx2:_negative(dx1),
                                dy2:_negative(dy1)]
                diff_sq = (feat_t1 - feat_t2) ** 2
                exp_diff_sq = torch.exp(torch.sum(-0.5 * diff_sq * theta, dim=1))
                gaussian[:, dx+self.span, dy+self.span,
                         dx2:_negative(dx1), dy2:_negative(dy1)] = exp_diff_sq

        # gaussian : (B, 1, K_h, K_w, H_s, W_s)
        gaussian = gaussian.view(feats.shape[0], 1,
                                 self.kernel_size, self.kernel_size,
                                 *down_image_size)

        return gaussian

    def forward(self, image, unary):
        ''' The forward propagation
        :param image: torch.Tensor, (B, C_img, H, W)
        :param unary: torch.Tensor, (B, C_cls, H, W)
        :param spacing: torch.Tensor, (B, 3)
        :param downsample_rate: float
        :return: prediction: torch.Tensor, (B, C_cls, H, W)
        '''
        # downsampling image
        down_image, down_image_size, pad_0, pad_1 = self._downsample(image)
        # generating features based on down_image and spacing
        feats = [self._generate_features(down_image, mode=mode) for mode in self.modes]
        # constructing gaussian kernels based on features
        gaussian_kernels = [self._generate_convolutional_filters(feats, down_image_size, eval('self.{}_theta'.format(mode)))
                            for i, (feats, mode) in enumerate(zip(feats, self.modes))]
        # calculating the normalized tensor of the gaussian kernels
        units = [torch.ones([1, 1, *down_image_size]).to(self.device)] * len(gaussian_kernels)
        norms = [1. / torch.sqrt(self._convolution(unit, kernel)+1e-20) for i, (unit, kernel) in enumerate(zip(units, gaussian_kernels))]

        # mean field approximation
        prediction = F.log_softmax(unary, dim=1)
        for iter_idx in range(self.num_iters):
            # downsampling prediction
            down_input, down_unary_size, pad_0, pad_1 = self._downsample(prediction)
            # message passing
            down_message = 0
            for i, (kernel, norm, mode) in enumerate(zip(gaussian_kernels, norms, self.modes)):
                iter_message = self._convolution(down_input, kernel * norm)
                down_message += (iter_message * eval('self.{}_weight'.format(mode)))
            # upsampling
            message = self._upsample(down_message, image.shape[-2:], pad_0, pad_1)
            # adding unary potentials
            prediction = self.momentum * prediction + (1 - self.momentum) * message

        # normalizing prediction
        prediction = F.log_softmax(prediction, dim=1)

        return prediction
