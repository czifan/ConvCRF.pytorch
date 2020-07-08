# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch
import numpy as np
from networks.convcrf2d import ConvCRF2d
from networks.unet import UNet
from networks.diceloss import BinaryDiceLoss
from torch.utils.data import Dataset
from glob import glob
from imutils import rotate
from imutils import resize
from utils.utils import save_checkpoint
from utils.utils import dice_coef_theoretical
from configs.config2d import config

class ImageDataset(Dataset):
    def __init__(self, model, split='train', crop_size=(512, 512)):
        super().__init__()
        self.model = model
        self.model.eval()
        self.split = split
        self.crop_size = crop_size
        self.image_files = glob(os.path.join(data_dir, split, 'image', '*.png'))
        self.label_files = glob(os.path.join(data_dir, split, 'label', '*.png'))
        print('[{}] Loaded {} samples'.format(split, self.__len__()))

    def _normalize(self, x, mean=128., std=128.):
        return (x - mean) / std

    def _random_crop(self, x, y, crop_size):
        H, W = x.shape
        pad = [[crop_size[0], crop_size[0]], [crop_size[1], crop_size[1]]]
        x = np.pad(x, pad, mode='constant', constant_values=0)
        y = np.pad(y, pad, mode='constant', constant_values=0)
        cx, cy = np.random.randint(0, H), np.random.randint(0, W)
        x = x[cx + crop_size[0] // 2:cx + crop_size[0] * 3 // 2, cy + crop_size[1] // 2:cy + crop_size[1] * 3 // 2]
        y = y[cx + crop_size[0] // 2:cx + crop_size[0] * 3 // 2, cy + crop_size[1] // 2:cy + crop_size[1] * 3 // 2]
        return x, y

    def __getitem__(self, item):

        def augmentation(x):
            try:
                x = rotate(x, rotate_angle)
            except:
                return
            H, W = x.shape[:2]
            x = resize(x, height=int(H * resize_ratio_h), width=int(W * resize_ratio_w))
            return x

        if self.split == 'train':
            rotate_angle = np.random.randint(-45, 45)
            resize_ratio_h = np.random.uniform(0.7, 1.3)
            resize_ratio_w = np.random.uniform(0.7, 1.3)
        else:
            rotate_angle = 0
            resize_ratio_h = 1.0
            resize_ratio_w = 1.0

        image_file = self.image_files[item]
        label_file = self.label_files[item]
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        image = augmentation(image)
        label = augmentation(label)
        if self.split == 'train':
            image, label = self._random_crop(image, label, self.crop_size)
        image = torch.from_numpy(image).unsqueeze(dim=0)
        unary = self.model(image.unsqueeze(dim=0))[0]
        label = torch.from_numpy(label).unsqueeze(dim=0)

        return image, unary, label

    def __len__(self):
        return len(self.image_files)

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    num_samples = len(train_loader)
    epoch_records = None
    for sample_idx, (images, unarys, labels) in enumerate(train_loader):
        images = images.float().to(device)
        unarys = unarys.float().to(device)
        labels = labels.float().to(device)
        outputs = model(images, unarys)
        loss, info = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch_records is None:
            epoch_records = {'unary_dice': []}
            for key, _ in info.items(): epoch_records[key] = []
        for key, value in info.items():
            epoch_records[key].append(value)
        epoch_records['unary_dice'].append(dice_coef_theoretical(unarys, labels))

        if (sample_idx and sample_idx % display == 0) or sample_idx == num_samples - 1:
            context = 'EP:{:03d}\tTI:{:03d}/{:03d}'.format(epoch, sample_idx, num_samples)
            context_info = '\t'
            for key, value in epoch_records.items():
                context_info += '{}:{:4f}({:.4f})\t'.format(key, value[-1], np.mean(value))
            print(context + context_info)

    epoch_records = {key:np.mean(value) for key, value in epoch_records.items()}
    return epoch_records

def valid(epoch, model, valid_loader, criterion):
    model.eval()
    num_samples = len(valid_loader)
    epoch_records = None
    for sample_idx, (images, unarys, labels) in enumerate(valid_loader):
        with torch.no_grad():
            images = images.float().to(device)
            unarys = unarys.float().to(device)
            labels = labels.float().to(device)
            outputs = model(images, unarys)
            loss, info = criterion(outputs, labels)

            if epoch_records is None:
                epoch_records = {'unary_dice': []}
                for key, _ in info.items(): epoch_records[key] = []
            for key, value in info.items():
                epoch_records[key].append(value)
            epoch_records['unary_dice'].append(dice_coef_theoretical(unarys, labels))

            if (sample_idx and sample_idx % display == 0) or sample_idx == num_samples - 1:
                context = '[V] EP:{:03d}\tTI:{:03d}/{:03d}'.format(epoch, sample_idx, num_samples)
                context_info = '\t'
                for key, value in epoch_records.items():
                    context_info += '{}:{:4f}({:.4f})\t'.format(key, value[-1], np.mean(value))
                print(context + context_info)

    print('=' * 80)

    epoch_records = {key:np.mean(value) for key, value in epoch_records.items()}
    return epoch_records


pretrained_file = 'pretrained/unet.pth.tar'
data_dir = 'data'
train_batch_size = 4
valid_batch_size = 2 * train_batch_size
lr = 1e-3
epochs = 20
display = 10
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    num_workers = 4
else:
    num_workers = 0

def main():
    model = UNet().to(device)
    model.load_state_dict(torch.load(pretrained_file, map_location=device)['state_dict'])
    model_crf = ConvCRF2d(config, kernel_size=5).to(device)
    criterion = BinaryDiceLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = ImageDataset(model, split='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        num_workers=num_workers, shuffle=True, pin_memory=True)
    valid_dataset = ImageDataset(model, split='valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size,
        num_workers=num_workers, shuffle=False, pin_memory=True)

    for epoch in range(epochs):
        train(epoch, model_crf, train_loader, criterion, optimizer)
        valid(epoch, model_crf, valid_loader, criterion)

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_file='pretrained/convcrf2d.pth.tar', is_best=False)

if __name__ == '__main__':
    main()


