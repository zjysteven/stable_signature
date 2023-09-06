# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional

import kornia.augmentation as K
from kornia.augmentation import AugmentationBase2D
import skimage as sk

import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffJPEG(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality
    
    def forward(self, x):
        with torch.no_grad():
            img_clip = utils_img.clamp_pixel(x)
            img_jpeg = utils_img.jpeg_compress(img_clip, self.quality)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x + img_gap
        return img_aug
    

class DiffImpulse(nn.Module):
    def __init__(self, scale=160):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        with torch.no_grad():
            imgs_clip = utils_img.unnormalize_img(x).clamp(0, 1)
            imgs_impulse = np.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
            for ii, img_clip in enumerate(imgs_clip):
                img_np = np.array(functional.to_pil_image(img_clip)) / 255
                img_impulse = np.clip(
                    sk.util.random_noise(img_np, mode="s&p", amount=self.scale), 0, 1
                )
                imgs_impulse[ii] = img_impulse
            imgs_impulse = torch.from_numpy(imgs_impulse).permute(0, 3, 1, 2).float().to(x.device)
            imgs_gap = utils_img.normalize_img(imgs_impulse) - x
            imgs_gap = imgs_gap.detach()
        imgs_aug = x + imgs_gap
        return imgs_aug
    

class DiffGaussian(nn.Module):
    def __init__(self, scale=0.09):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x + torch.randn_like(x.detach()) * self.scale
    

class DiffShot(nn.Module):
    def __init__(self, scale=160):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        with torch.no_grad():
            imgs_clip = utils_img.unnormalize_img(x).clamp(0, 1)
            imgs_shot = np.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
            for ii, img_clip in enumerate(imgs_clip):
                img_np = np.array(functional.to_pil_image(img_clip)) / 255
                img_shot = np.clip(
                    np.random.poisson(img_np * self.scale) / self.scale, 0, 1
                )
                imgs_shot[ii] = img_shot
            imgs_shot = torch.from_numpy(imgs_shot).permute(0, 3, 1, 2).float().to(x.device)
            imgs_gap = utils_img.normalize_img(imgs_shot) - x
            imgs_gap = imgs_gap.detach()
        imgs_aug = x + imgs_gap
        return imgs_aug


class DiffSpeckle(nn.Module):
    def __init__(self, scale=0.05):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x + (x * torch.randn_like(x) * self.scale).detach()


class DiffMixup(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        index = torch.randperm(x.size(0)).to(x.device)
        img1, img2 = x, x[index]
        return self.alpha * img1 + (1 - self.alpha) * img2.detach()


class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=10, high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in range(low,high,10)]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output


class HiddenAug(nn.Module):
    """Dropout p = 0.3,Dropout p = 0.7, Cropout p = 0.3, Cropout p = 0.7, Crop p = 0.3, Crop p = 0.7, Gaussian blur σ = 2, Gaussian blur σ = 4, JPEG-drop, JPEG-mask and the Identity layer"""
    def __init__(self, img_size, p_crop=0.3, p_blur=0.3, p_jpeg=0.3, p_rot=0.3, p_color_jitter=0.3, p_res=0.3):
        super().__init__()
        augmentations = []
        hflip = K.RandomHorizontalFlip(p=1)
        augmentations += [nn.Identity(), hflip]
        if p_crop > 0:
            crop1 = int(img_size * np.sqrt(0.3))
            crop2 = int(img_size * np.sqrt(0.7))
            crop1 = K.RandomCrop(size=(crop1, crop1), p=1) # Crop 0.3   
            crop2 = K.RandomCrop(size=(crop2, crop2), p=1) # Crop 0.7       
            augmentations += [crop1, crop2]
        if p_res > 0:
            res1 = int(img_size * np.sqrt(0.3))
            res2 = int(img_size * np.sqrt(0.7))
            res1 = K.RandomResizedCrop(size=(res1, res1), scale=(1.0,1.0), p=1) # Resize 0.3
            res2 = K.RandomResizedCrop(size=(res2, res2), scale=(1.0,1.0), p=1) # Resize 0.7
            augmentations += [res1, res2]
        if p_blur > 0:
            blur1 = K.RandomGaussianBlur(kernel_size=(11,11), sigma= (2.0, 2.0), p=1) # Gaussian blur σ = 2
            # blur2 = K.RandomGaussianBlur(kernel_size=(25,25), sigma= (4.0, 4.0), p=1) # Gaussian blur σ = 4
            augmentations += [blur1]
            # augmentations += [blur1, blur2]
        if p_jpeg > 0:
            diff_jpeg1 = DiffJPEG(quality=50)  # JPEG50
            diff_jpeg2 = DiffJPEG(quality=80)  # JPEG80
            augmentations += [diff_jpeg1, diff_jpeg2]
        if p_rot > 0:
            # aff1 = K.RandomAffine(degrees=(-10,10), p=1)
            # aff2 = K.RandomAffine(degrees=(90,90), p=1)
            # aff3 = K.RandomAffine(degrees=(-90,-90), p=1)
            # augmentations += [aff1]
            # augmentations += [aff2, aff3]
            augmentations += [K.RandomRotation(degrees=(-90,90), p=1)]
        if p_color_jitter > 0:
            jitter1 = K.ColorJiggle(brightness=(1.5, 1.5), contrast=0, saturation=0, hue=0, p=1)
            jitter2 = K.ColorJiggle(brightness=0, contrast=(1.5, 1.5), saturation=0, hue=0, p=1)
            jitter3 = K.ColorJiggle(brightness=0, contrast=0, saturation=(1.5,1.5), hue=0, p=1)
            jitter4 = K.ColorJiggle(brightness=0, contrast=0, saturation=0, hue=(0.25, 0.25), p=1)
            augmentations += [jitter1, jitter2, jitter3, jitter4]
        self.hidden_aug = K.AugmentationSequential(*augmentations, random_apply=1).to(device)
        
    def forward(self, x):
        return self.hidden_aug(x)
    

class HiddenAug_v1(nn.Module):
    def __init__(self, img_size, p_noisy=0.5, aug_order='parallel', noisy_list=['jpeg']):
        super().__init__()
        size1 = int(img_size * np.sqrt(0.3))
        size2 = int(img_size * np.sqrt(0.7))
        self.basic_list = [
            nn.Identity(),
            K.RandomHorizontalFlip(p=1),
            K.RandomCrop(size=(size1, size1), p=1),
            K.RandomCrop(size=(size2, size2), p=1),
            K.RandomResizedCrop(size=(size1, size1), scale=(1.0, 1.0), p=1),
            K.RandomResizedCrop(size=(size2, size2), scale=(1.0, 1.0), p=1),
            K.RandomRotation(degrees=(-90, 90), p=1)
        ]
        self.jpeg = [
            DiffJPEG(quality=50),
            DiffJPEG(quality=80)
        ]
        self.speckle = [
            DiffSpeckle(scale=0.05),
            DiffSpeckle(scale=0.10)
        ]
        self.shot = [
            DiffShot(scale=1000),
            DiffShot(scale=5000)
        ]
        self.mixup = [
            DiffMixup(alpha=0.9),
            DiffMixup(alpha=0.95)
        ]
        self.gaussian = [
            DiffGaussian(scale=0.06),
            DiffGaussian(scale=0.10)
        ]
        
        self.noisy_list = noisy_list
        self.p_noisy = p_noisy
        self.aug_order = aug_order
    
    def forward(self, x):
        if self.aug_order == 'parallel':
            if np.random.rand() < 1 - self.p_noisy:
                aug = np.random.choice(self.basic_list)
                return aug(x)
            else:
                aug_type = np.random.choice(self.noisy_list)
                aug = np.random.choice(getattr(self, aug_type))
                return aug(x)
        elif self.aug_order == 'sequential':
            raise NotImplementedError
        

class KorniaAug(nn.Module):
    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3/4, 4/3), blur_size=17, color_jitter=(1.0, 1.0, 1.0, 0.3), diff_jpeg=10,
                p_crop=0.5, p_aff=0.5, p_blur=0.5, p_color_jitter=0.5, p_diff_jpeg=0.5, 
                cropping_mode='slice', img_size=224
            ):
        super(KorniaAug, self).__init__()
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter).to(device)
        # self.jitter = K.RandomPlanckianJitter(p=p_color_jitter).to(device)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop(size=(img_size,img_size),scale=crop_scale,ratio=crop_ratio, p=p_crop, cropping_mode=cropping_mode).to(device)
        self.hflip = K.RandomHorizontalFlip().to(device)
        self.blur = RandomBlur(blur_size, p_blur).to(device)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg).to(device)
    
    def forward(self, input):
        input = self.diff_jpeg(input)
        input = self.aff(input)
        input = self.crop(input)
        input = self.blur(input)
        input = self.jitter(input)
        input = self.hflip(input)
        return input
