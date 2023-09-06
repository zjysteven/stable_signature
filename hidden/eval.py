import argparse
import datetime
import json
import os
import time
import random
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import functional
from torchvision.utils import save_image

import data_augmentation
import utils
from utils_img import psnr, normalize_img, unnormalize_img
import models
import attenuations
from attacks import attack_modules

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--train_dir", type=str, default="/checkpoint/pfz/watermarking/data/train_coco_10k_orig/0")
    aa("--val_dir", type=str, default="/checkpoint/pfz/watermarking/data/coco_1k_orig/0")
    aa("--model_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")

    group = parser.add_argument_group('Marking parameters')
    aa("--num_bits", type=int, default=32, help="Number of bits of the watermark (Default: 32)")
    aa("--redundancy", type=int, default=1, help="Redundancy of the watermark (Default: 1)")
    aa("--img_size", type=int, default=128, help="Image size")

    group = parser.add_argument_group('Encoder parameters')
    aa("--encoder", type=str, default="hidden", help="Encoder type (Default: hidden)")
    aa('--encoder_depth', default=4, type=int, help='Number of blocks in the encoder.')
    aa('--encoder_channels', default=64, type=int, help='Number of channels in the encoder.')
    aa("--use_tanh", type=utils.bool_inst, default=True, help="Use tanh scaling. (Default: True)")
    aa("--encoder_use_skip", type=utils.bool_inst, default=False, help="Use skip connections. (Default: False)")

    group = parser.add_argument_group('Decoder parameters')
    aa("--decoder", type=str, default="hidden", help="Decoder type (Default: hidden)")
    aa("--decoder_depth", type=int, default=8, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_channels", type=int, default=64, help="Number of blocks in the decoder (Default: 4)")
    aa("--decoder_use_skip", type=utils.bool_inst, default=False, help="Use skip connections. (Default: False)")

    group = parser.add_argument_group('Training parameters')
    aa("--scaling_w", type=float, default=1.0, help="Scaling of the watermark signal. (Default: 1.0)")
    aa("--scaling_i", type=float, default=1.0, help="Scaling of the original image. (Default: 1.0)")

    group = parser.add_argument_group('Loader parameters')
    aa("--batch_size", type=int, default=16, help="Batch size. (Default: 16)")
    aa("--workers", type=int, default=8, help="Number of workers for data loading. (Default: 8)")

    group = parser.add_argument_group('Attenuation parameters')
    aa("--attenuation", type=str, default=None, help="Attenuation type. (Default: jnd)")
    aa("--scale_channels", type=utils.bool_inst, default=True, help="Use channel scaling. (Default: True)")

    group = parser.add_argument_group('Evaluation parameters')
    aa("--use_atten", type=utils.bool_inst, default=False, help="Use attenuation. (Default: False)")
    aa("--num_keys", type=int, default=1)
    aa("--seed", default=0, type=int, help='Random seed')
    aa("--whiten", type=utils.bool_inst, default=True, help="Whiten the decoder. (Default: True)")

    return parser

def main(params):
    # Set number of threads
    torch.set_num_threads(params.workers)

    # load trained model
    encoder = models.HiddenEncoder(
        num_blocks=params.encoder_depth, 
        num_bits=params.num_bits, 
        channels=params.encoder_channels, 
        last_tanh=params.use_tanh,
        skip=params.encoder_use_skip
    )

    ckpt_path = os.path.join(params.model_dir, 'checkpoints', 'checkpoint.pth')
    state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
    encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval().to(device)

    if params.whiten:
        if not os.path.exists(os.path.join(params.model_dir, 'checkpoints', 'decoder_whit.pth')):
            if params.decoder_use_skip:
                decoder = models.HiddenDecoderSkip(
                    num_blocks=params.decoder_depth, 
                    num_bits=params.num_bits * params.redundancy, 
                    channels=params.decoder_channels
                )
            else:
                decoder = models.HiddenDecoder(
                    num_blocks=params.decoder_depth, 
                    num_bits=params.num_bits * params.redundancy, 
                    channels=params.decoder_channels
                )
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
            print(decoder.load_state_dict(decoder_state_dict, strict=False))
            decoder.to(device)
            decoder.eval()

            # whiten the decoder
            with torch.no_grad():
                # features from the dataset
                transform = transforms.Compose([
                    transforms.Resize(params.img_size),
                    transforms.CenterCrop(params.img_size),
                    transforms.ToTensor(),
                    normalize_img,
                ])
                loader = utils.get_dataloader(params.train_dir, transform, batch_size=params.batch_size, num_workers=4, drop_last=False, shuffle=False)
                ys = []
                for i, (x, _) in tqdm(enumerate(loader), total=len(loader), leave=True, desc='Whitening...'):
                    y = decoder(x.to(device))
                    ys.append(y.cpu())
                ys = torch.cat(ys, dim=0)
                nbit = ys.shape[1]
                print(ys.shape)
                
                # whitening
                mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
                ys_centered = ys - mean # NxD
                cov = ys_centered.T @ ys_centered
                e, v = torch.linalg.eigh(cov)
                L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
                #weight = torch.mm(L, v.T)
                weight = torch.mm(torch.mm(v, L), v.T)
                bias = -torch.mm(mean, weight.T).squeeze(0)
                linear = nn.Linear(nbit, nbit, bias=True)
                linear.weight.data = np.sqrt(nbit) * weight
                linear.bias.data = np.sqrt(nbit) * bias
                decoder = nn.Sequential(decoder, linear.to(device))
                decoder.eval()
                torchscript_m = torch.jit.script(decoder)
                saving_path = ckpt_path.replace('checkpoint.pth', 'decoder_whit.pth')
                print(f'>>> Saving whitened decoder at {saving_path}...')
                torch.jit.save(torchscript_m, saving_path)
        else:
            decoder = torch.jit.load(os.path.join(params.model_dir, 'checkpoints', 'decoder_whit.pth')).to(device)
            decoder.eval()
    else:
        decoder = models.HiddenDecoder(
            num_blocks=params.decoder_depth, 
            num_bits=params.num_bits * params.redundancy, 
            channels=params.decoder_channels
        )
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}
        decoder.load_state_dict(decoder_state_dict)
        decoder.eval().to(device)

    attenuation = attenuations.JND(preprocess=unnormalize_img).to(device)

    # some setup for attack modules
    attack_modules['resized_crop'].transform.size = params.img_size
    attack_list = [
        "clean",
        "color_jitter",
        "center_crop_0.3",
        "resize_0.3",
        "resized_crop",
        "rotate",
        "jpeg_compress_50",
        "colorize",
        "motion_deblur_11",
        "gaussian_denoise",
        "auto_encode_4"
    ]
    for am in attack_list:
        if am != 'clean':
            attack_modules[am].setup()

    # eval data loading
    val_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        normalize_img,
    ])
    val_loader = utils.get_dataloader(params.val_dir, transform=val_transform, batch_size=params.batch_size, num_workers=4, drop_last=False, shuffle=False)

    # Set seeds for reproductibility 
    seed = params.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # evaluation loop
    with torch.no_grad():
        metric_logger = utils.MetricLogger(delimiter="  ")
        # for key_id in tqdm(range(params.num_keys), total=params.num_keys, leave=True):
        #     msgs = torch.rand((1, params.num_bits)) > 0.5
        #     msgs = 2 * msgs.type(torch.float).to(device) - 1

        #     for it, (imgs, _) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        #         imgs = imgs.to(device, non_blocking=True)
        #         bs = imgs.shape[0]

        #         deltas_w = encoder(imgs, msgs.repeat(bs, 1))
        #         if params.use_atten:
        #             heatmaps = attenuation.heatmaps(imgs) # b 1 h w
        #             deltas_w = deltas_w * heatmaps
        #         imgs_w = params.scaling_i * imgs + params.scaling_w * deltas_w

        #         psnrs = psnr(imgs_w, imgs)
        #         log_stats = {'psnr': torch.mean(psnrs).item()}

        #         for name in attack_list:
        #             if name == 'clean':
        #                 eval_aug = nn.Identity()
        #             else:
        #                 eval_aug = attack_modules[name]
                    
        #             imgs_aug = normalize_img(eval_aug(unnormalize_img(imgs_w).clamp(0, 1)))
        #             fts = decoder(imgs_aug)
                
        #             decoded_msgs = torch.sign(fts) > 0
        #             diff = (~torch.logical_xor(msgs.repeat(bs, 1)>0, decoded_msgs))
        #             log_stats[f'bit_acc_{name}'] = diff.float().mean().item()
                
        #         for name, value in log_stats.items():
        #             metric_logger.update(**{name: value})
        
        for it, (imgs, _) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            imgs = imgs.to(device, non_blocking=True)
            bs = imgs.shape[0]
            msgs = torch.rand((bs, params.num_bits)) > 0.5
            msgs = 2 * msgs.type(torch.float).to(device) - 1

            deltas_w = encoder(imgs, msgs)
            if params.use_atten:
                heatmaps = attenuation.heatmaps(imgs) # b 1 h w
                deltas_w = deltas_w * heatmaps
            imgs_w = params.scaling_i * imgs + params.scaling_w * deltas_w

            psnrs = psnr(imgs_w, imgs)
            log_stats = {'psnr': torch.mean(psnrs).item()}

            for name in attack_list:
                if name == 'clean':
                    eval_aug = nn.Identity()
                else:
                    eval_aug = attack_modules[name]
                
                imgs_aug = normalize_img(eval_aug(unnormalize_img(imgs_w).clamp(0, 1)))
                fts = decoder(imgs_aug)
            
                decoded_msgs = torch.sign(fts) > 0
                diff = (~torch.logical_xor(msgs>0, decoded_msgs))
                log_stats[f'bit_acc_{name}'] = diff.float().mean().item()
            
            for name, value in log_stats.items():
                metric_logger.update(**{name: value})

    final_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    df = pd.DataFrame(final_stats, index=[0]).T
    df.to_csv(os.path.join(params.model_dir, f'eval_alpha={params.scaling_w}_atten={params.use_atten}.csv'), header=False)
    print(df)


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)