# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import shutil
import skimage
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from utils.pascal_visualizer import PascalVisualizer

def dice_coef_theoretical(outputs, targets):
    if not isinstance(outputs, np.ndarray):
        outputs = outputs.detach().cpu().numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.detach().cpu().numpy()
    outputs = (outputs >= 0.5).astype(np.float)
    targets = (targets >= 1).astype(np.float)
    inter = (outputs * targets).sum()
    dice = 2. * inter * 1.0 / (outputs.sum() + targets.sum() + 1e-20) * 1.0
    return dice

def process_img_unary(img, unary):
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    unary = unary.transpose(2, 0, 1)[np.newaxis, ...]
    return img, unary

def plot_results(image, unary, prediction, label):
    myvis = PascalVisualizer()
    coloured_label = myvis.id2color(id_image=label)

    unary_hard = np.argmax(unary, axis=2)
    coloured_unary = myvis.id2color(id_image=unary_hard)

    prediction_hard = np.argmax(prediction[0], axis=0)
    coloured_crf = myvis.id2color(id_image=prediction_hard)

    _, axarr = plt.subplots(1, 4, figsize=(20, 5))
    axarr[0].imshow(image)
    axarr[0].set_title('image')
    axarr[1].imshow(coloured_label)
    axarr[1].set_title('label')
    axarr[2].imshow(coloured_unary)
    axarr[2].set_title('unary')
    axarr[3].imshow(coloured_crf)
    axarr[3].set_title('crf')
    plt.show()

def augment_label(label, num_classes, scale=8, keep_prop=0.8):
    ''' Add noise to label for synthetic benchmark.
        reference: https://github.com/MarvinTeichmann/ConvCRF '''

    def _onehot(label, numclasses):
        return np.eye(num_classes)[label]

    shape = label.shape
    label = label.reshape(shape[0], shape[1])

    onehot = _onehot(label, num_classes)
    lower_shape = (shape[0] // scale, shape[1] // scale)

    label_down = skimage.transform.resize(
        onehot, (lower_shape[0], lower_shape[1], num_classes),
        order=1, preserve_range=True, mode='constant')

    onehot = skimage.transform.resize(label_down,
                                      (shape[0], shape[1], num_classes),
                                      order=1, preserve_range=True,
                                      mode='constant')

    noise = np.random.randint(0, num_classes, lower_shape)
    noise = _onehot(noise, num_classes)
    noise_up = skimage.transform.resize(noise,
                                        (shape[0], shape[1], num_classes),
                                        order=1, preserve_range=True,
                                        mode='constant')

    mask = np.floor(keep_prop + np.random.rand(*lower_shape))
    mask_up = skimage.transform.resize(mask, (shape[0], shape[1], 1),
                                       order=1, preserve_range=True,
                                       mode='constant')

    noised_label = mask_up * onehot + (1 - mask_up) * noise_up

    return noised_label

def save_checkpoint(states, save_file, is_best):
    torch.save(states, save_file)
    if is_best:
        shutil.copy(save_file, os.path.join(*save_file.split('/')[:-1], 'best.pth.tar'))