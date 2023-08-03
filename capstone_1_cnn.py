# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:16:47 2023

@author: Nikhil
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from torch.nn.functional import relu
import glob
import os
import random
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
import numpy as np           
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) 
        contracting_12_out = self.contracting_12(contracting_11_out) 
        contracting_21_out = self.contracting_21(contracting_12_out) 
        contracting_22_out = self.contracting_22(contracting_21_out) 
        contracting_31_out = self.contracting_31(contracting_22_out) 
        contracting_32_out = self.contracting_32(contracting_31_out) 
        contracting_41_out = self.contracting_41(contracting_32_out) 
        contracting_42_out = self.contracting_42(contracting_41_out) 
        middle_out = self.middle(contracting_42_out) 
        expansive_11_out = self.expansive_11(middle_out) 
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) 
        expansive_21_out = self.expansive_21(expansive_12_out) 
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) 
        expansive_31_out = self.expansive_31(expansive_22_out) 
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) 
        expansive_41_out = self.expansive_41(expansive_32_out) 
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) 
        output_out = self.output(expansive_42_out) 
        return output_out
 
# padding image
def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img

# data loader
class RainDataset(Dataset):
    def __init__(self, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = 'rain100L', data_type, patch_size
        self.rain_images = sorted(glob.glob('rain100L/rain100L/train/rain/*.png'))
        self.norain_images = sorted(glob.glob('rain100L/rain100L/train/norain/*.png'))
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]))
        norain = T.to_tensor(Image.open(self.norain_images[idx % self.num]))
        h, w = rain.shape[1:]

        if self.data_type == 'train':
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        else:
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
        return rain, norain, image_name, h, w
        

def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()
    
def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)

def test_loop(net, data_loader, num_iter):
    intermediate_tracked_images = ['norain-11.png', 'norain-32.png', 'norain-40.png', 'norain-48.png', 'norain-53.png', 'norain-83.png']
    intermediate_tracked_iters = [1000, 2000, 3000, 4000, 5000, 10000, 15000]
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.cuda(), norain.cuda()
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = 'result_capstone/rain100L/{}'.format(name[0])

            # saving intermediate images
            if num_iter in intermediate_tracked_iters:
                if name[0] in intermediate_tracked_images:
                    this_psnr = round(total_psnr / count, 2)
                    img_file_name = f"{name[0].split('.')[0]}_psnr_{this_psnr}.{name[0].split('.')[1]}"
                    this_inter_save_path = f'result/intermediate_train_images/{img_file_name}'
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))                
                    Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(this_inter_save_path)

            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 16000, total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count


def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format('result_capstone', 'rain100L'), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format('result_capstone', 'rain100L'), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format('result_capstone', 'rain100L'))


# main
if __name__ == '__main__':
    test_dataset = RainDataset('test', patch_size=224)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
    model = UNet(3).cuda()
    
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=16000, eta_min=1e-6)
    total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
    train_bar = tqdm(range(1, 16000 + 1), initial=1, dynamic_ncols=True)
    milestone = [5000, 7000, 10000, 12000, 15000]
    batch_size = [4,4,4,4,4]
    for n_iter in train_bar:
        # progressive learning
        if n_iter == 1 or n_iter - 1 in milestone:
            end_iter = milestone[i] if i < len(milestone) else 16000
            start_iter = milestone[i - 1] if i > 0 else 0
            lengthn = batch_size[i] * (end_iter - start_iter)
            train_dataset = RainDataset(data_type='train', patch_size=224, length=lengthn)
            #train_dataset = RainDataset(data_type='train',patch_size=224, length)
            num = 15
            #a_org, b_org, _, _, _ = test_dataset.__getitem__(num)
            train_loader = iter(DataLoader(train_dataset, batch_size[i], True, num_workers=1))
            
            i += 1
        # train
        model.train()
        rain, norain, name, h, w = next(train_loader)
        rain, norain = rain.cuda(), norain.cuda()
        out = model(rain)
        loss = F.l1_loss(out, norain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += rain.size(0)
        total_loss += loss.item() * rain.size(0)
        train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'.format(n_iter, 16000, total_loss / total_num))

        lr_scheduler.step()
        if n_iter % 1000 == 0:
            results['Loss'].append('{:.3f}'.format(total_loss / total_num))
            save_loop(model, test_loader, n_iter)

























# testing image getting loaded
data = RainDataset(data_type='train', patch_size=224)
data.__len__()
data.num
data.__getitem__(0)


# image created
img = read_image('rain100L_new/rain100L/train/rain/norain-1.png')     
img = img.float()
img = img[None, :]

a = Encoder_Decoder(3)
b = a.forward(img)

#  testing

test = torch.rand(1, 3, 224, 224)

img.shape

b.shape





























########## Rough Work #################################

import numpy as np
import pandas as pd

# dataframe
df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))

# standard deviation
df.std()

# mean
df.mean()












