# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import itertools
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class ConvCRF3d(nn.Module):
    def __init__(self, config, kernel_size):
        super().__init__()

        # fixed parameters
        self.num_classes = config.num_classes
        self.kernel_size = kernel_size
        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size] * 3
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1 and self.kernel_size[2] % 2 == 1, "'kernel_size' should be odd"
        self.span = [self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[2] // 2]
        self.device = config.device
        self.downsample_rate = config.downsample_rate # blur
        self.num_iters = config.num_iters # the maximum iterations number of mean field approximation
        self.momentum = config.momentum
        self.modes = config.modes # gaussian kernel feature construction mode: pos (position) | col (colour) | com (combine)
        self.channels = config.channels # the number of channels corresponding to different types of characteristics
        self.pos_scale = config.pos_scale
        self.col_scale = config.col_scale

        # learnable parameters
        self.thetas = [self.register_parameter('{}_theta'.format(mode), nn.Parameter(torch.ones(1, c, 1, 1, 1).to(config.device)))
                       for c, mode in zip(self.channels, self.modes)] # scaling parameters of different gaussian kernels
        self.weights = [self.register_parameter('{}_weight'.format(mode), nn.Parameter(torch.Tensor([1. / len(self.modes)]).to(config.device)))
                        for mode in self.modes] # weighting parameters of different gaussian kernels

        # compatibility transformation
        if config.compa == 'conv': # using 1x1 conv as compatibility transformation
            in_channels = self.num_classes * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            out_channels = in_channels
            self.compa = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
            self.compa.weight.data.fill_(1.)
        elif config.compa == 'potts': # using potts model
            self.compa = None

    def _create_mesh(self, image_size, spacing):
        ''' creating 3d mesh according to the spacing of image
            Note: the spacing of images except to CT images is set to (1., 1., 1.)
        :param image_size: np.array, (D, H, W)
        :param spacing: torch.Tensor, (3,)
        :return: 3d mesh: torch.Tensor, (3, D, H, W)
        '''
        mesh = np.array(list(itertools.product(*[np.array(range(image_size[i])) * spacing[i].item() for i in range(len(image_size))])))
        mesh = mesh.transpose(1, 0).reshape(len(spacing), *image_size)
        mesh = torch.from_numpy(mesh).float().to(self.device)
        return mesh

    def _generate_features(self, img, spacing, mode):
        ''' generating features
        :param img: torch.Tensor, (B, C_img, D, H, W)
        :param spacing: torch.Tensor, (B, 3)
        :param mode: str, gaussian kernel feature construction mode: pos (position) | col (colour) | com (combine)
        :return: features: torch.Tensor, (B, C_features, D, H, W)
        '''
        if mode == 'pos':
            return torch.stack([self._create_mesh(img.shape[-3:], spacing[b]) for b in range(img.shape[0])]) / self.pos_scale
        elif mode == 'col':
            return img / self.col_scale
        elif mode == 'com':
            return torch.cat([torch.stack([self._create_mesh(img.shape[-3:], spacing[b]) for b in range(img.shape[0])]) / self.pos_scale,
                                           img / self.col_scale], dim=1)
        else:
            print("'{}' is a no defined mode.".format(mode))
            return None

    def _downsample(self, input):
        ''' Downsampling is performed to reduce the computation and expand the receptive field
        :param input: torch.Tensor, (B, C_input, D, H, W)
        :return: down_input: torch.Tensor, (B, C_input, D_s, H_s, W_s)
                 down_image_size: list, (D_s, H_s, W_s)
                 pad_0: float
                 pad_1: float
                 pad_2: float
        '''
        image_size = input.shape[-3:]
        if self.downsample_rate > 1:
            off_0 = (self.downsample_rate - image_size[0] % self.downsample_rate) % self.downsample_rate
            off_1 = (self.downsample_rate - image_size[1] % self.downsample_rate) % self.downsample_rate
            off_2 = (self.downsample_rate - image_size[2] % self.downsample_rate) % self.downsample_rate
            pad_0 = int(ceil(off_0 / 2))
            pad_1 = int(ceil(off_1 / 2))
            pad_2 = int(ceil(off_2 / 2))
            down_input = F.avg_pool3d(input, kernel_size=self.downsample_rate,
                                      padding=(pad_0, pad_1, pad_2), count_include_pad=False)
            down_image_size = [ceil(image_size[0] / self.downsample_rate),
                               ceil(image_size[1] / self.downsample_rate),
                               ceil(image_size[2] / self.downsample_rate)]
        else:
            pad_0 = 0
            pad_1 = 0
            pad_2 = 0
            down_image_size = image_size
            down_input = input

        return down_input, down_image_size, pad_0, pad_1, pad_2

    def _upsample(self, input, image_size, pad_0, pad_1, pad_2):
        ''' Upsampling to restore original size
        :param input: torch.Tensor, (D, C_input, D_s, H_s, W_s)
        :param image_size: list, (D, H, S)
        :param pad_0: float
        :param pad_1: float
        :param pad_2: float
        :return: input: torch.Tensor, (D, C_input, D, H, W)
        '''
        if self.downsample_rate > 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                input = F.upsample(input, scale_factor=self.downsample_rate, mode='trilinear')
            input = input[:, :,
                    pad_0:pad_0 + image_size[0],
                    pad_1:pad_1 + image_size[1],
                    pad_2:pad_2 + image_size[2]].contiguous()
        return input

    def _convolution(self, input, gaussian_kernel, compa=None):
        ''' Computing convolution operation
        :param input: torch.Tensor, (B, C_input, D_s, H_s, W_s)
        :param gaussian_kernel: torch.Tensor, (B, 1, K_d, K_h, K_w, D_s, H_s, W_s)
        :param compa: function or None, compatibility transformation
        :return: message, torch.Tensor, (B, C_cls, D_s, H_s, W_s)
        '''
        B, C, D, H, W = input.shape  # (B, C_input, D_s, H_s, W_s)
        input = F.pad(input, pad=[self.span[2]] * 2 + [self.span[1]] * 2 + [self.span[0]] * 2, mode='constant',
                      value=input.min().item())
        # input_col=input_unfold: (B, C_cls, K_d, K_h, K_w, D_s, H_s, W_s)
        input_unfold = input.unfold(2, self.kernel_size[0], 1).unfold(3, self.kernel_size[1], 1).unfold(4, self.kernel_size[2], 1)
        input_unfold = input_unfold.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous()
        input_col = input_unfold
        # product: (B, 1, K_d, K_h, K_w, D_s, H_s, W_s) * (B, C_cls, K_d, K_h, K_w, D_s, H_s, W_s) --> (B, C_cls, K_d, K_h, K_w, D_s, H_s, W_s)
        product = gaussian_kernel * input_col
        # compatibility transformation
        if compa is not None:
            product = compa(product.view(B, -1, D, H, W))
        # product: (B, C_cls, -1, D_s, H_s, W_s)
        product = product.view([B, C, self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], D, H, W])
        # message : (B, C_cls, D_s, H_s, W_s)
        message = product.sum(dim=2)

        return message

    def _generate_convolutional_filters(self, feats, down_image_size, theta):
        ''' Generating convolutional filter according features
        :param feats: torch.Tensor, (B, C_features, D_s, H_s, W_s)
        :param down_image_size: list, (D_s, H_s, W_s)
        :param theta: torch.nn.Parameter, scaling parameters
        :return: gaussian: torch.Tensor, (B, 1, K_d, K_h, K_w, D_s, H_s, W_s)
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

        # feats : (B, C_features, D_s, H_s, W_s)
        # gaussian : (B, K_d, K_h, K_w, D_s, H_s, W_s)
        gaussian = feats.data.new(feats.shape[0],
                                  self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                  *down_image_size).fill_(0).to(self.device)

        for dz in range(-self.span[0], self.span[0] + 1):
            for dy in range(-self.span[1], self.span[1] + 1):
                for dx in range(-self.span[2], self.span[2] + 1):
                    dz1, dz2 = _get_ind(dz)
                    dy1, dy2 = _get_ind(dy)
                    dx1, dx2 = _get_ind(dx)
                    feat_t1 = feats[:, :,
                              dz1:_negative(dz2),
                              dy1:_negative(dy2),
                              dx1:_negative(dx2)]
                    feat_t2 = feats[:, :,
                              dz2:_negative(dz1),
                              dy2:_negative(dy1),
                              dx2:_negative(dx1)]
                    diff_sq = (feat_t1 - feat_t2) ** 2
                    exp_diff_sq = torch.exp(torch.sum(-0.5 * diff_sq * theta, dim=1))
                    gaussian[:, dz + self.span[0], dy + self.span[1], dx + self.span[2],
                    dz2:_negative(dz1), dy2:_negative(dy1), dx2:_negative(dx1)] = exp_diff_sq

        # gaussian : (B, 1, K_d, K_h, K_w, D_s, H_s, W_s)
        gaussian = gaussian.view(feats.shape[0], 1,
                                 self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                 *down_image_size)

        return gaussian

    def forward(self, image, unary, spacing, downsample_rate=1):
        ''' The forward propagation
        :param image: torch.Tensor, (B, C_img, D, H, W)
        :param unary: torch.Tensor, (B, C_cls, D, H, W)
        :param spacing: torch.Tensor, (B, 3)
        :param downsample_rate: float
        :return: prediction: torch.Tensor, (B, C_cls, D, H, W)
        '''
        # downsampling image
        down_image, down_image_size, pad_0, pad_1, pad_2 = self._downsample(image)
        # generating features based on down_image and spacing
        feats = [self._generate_features(down_image, spacing=spacing, mode=mode) for mode in self.modes]
        # constructing gaussian kernels based on features
        gassiuan_kernels = [self._generate_convolutional_filters(feats, down_image_size, eval('self.{}_theta'.format(mode))) for
                            feats, mode in zip(feats, self.modes)]
        # calculating the normalized tensor of the gaussian kernels
        units = [torch.ones([1, 1, *down_image_size]).to(self.device)] * len(gassiuan_kernels)
        norms = [1. / (self._convolution(unit, kernel) + 1e-20) for unit, kernel in zip(units, gassiuan_kernels)]

        # mean field approximation
        basic_prediction = unary
        prediction = unary
        for iter_idx in range(self.num_iters):
            # downsampling prediction
            down_input, down_unary_size, pad_0, pad_1, pad_2 = self._downsample(prediction)
            # message passing
            down_message = 0
            for kernel, norm, mode in zip(gassiuan_kernels, norms, self.modes):
                iter_message = self._convolution(down_input, kernel, compa=self.compa) * norm
                down_message += (iter_message * eval('self.{}_weight'.format(mode)))
                del iter_message
            # upsampling
            message = self._upsample(down_message, image.shape[-3:], pad_0, pad_1, pad_2)
            # adding unary potentials
            prediction = self.momentum * prediction + (1 - self.momentum) * message
            del message, down_input
            # determining whether or not it converges
            if torch.mean(torch.abs(prediction - basic_prediction)) < 1e-2: break
            else: basic_prediction = prediction

        # normalizing prediction
        prediction = prediction / prediction.max()

        return prediction
