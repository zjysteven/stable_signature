import os

import PIL
import numpy as np
from skimage.util import random_noise
import torch
from torchvision.transforms import functional

from .base import BaseAttack
from .model_utils import MIRNet_v2


class GaussianDenoise(BaseAttack):
    """Apply a specific type of noise to the input images, and then remove the noise.

    Args:
        noise_type (str): type of noise to be applied. Should be one of 'gaussian', 'salt_and_pepper', or 'speckle'.
    """

    def __init__(self, remove=True):
        self._setup = False
        self._remove = remove

    def setup(self, local_rank=None):
        self._setup = True
        parameters = {
            "inp_channels": 3,
            "out_channels": 3,
            "n_feat": 80,
            "chan_factor": 1.5,
            "n_RRG": 4,
            "n_MRB": 2,
            "height": 3,
            "width": 2,
            "scale": 1,
        }
        url = "https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth"
        self.model = MIRNet_v2(
            **parameters
        )  # as there are many instantiated attacks, do not load the model onto the aceelerator until attack, in order to save GPU memory
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(url)["params"])
        self.model.eval().cuda()
        if local_rank is not None:  # DDP
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )

    def attack(self, x):
        if not self._setup:
            self.setup()
        img_aug = torch.zeros_like(x)
        for i, img in enumerate(x):
            pil_img = functional.to_pil_image(img)
            img = self.add_noise(pil_img)
            if self._remove:
                img = self.remove_noise(img)
            if isinstance(img, PIL.Image.Image):
                img = functional.to_tensor(img)
            img_aug[i] = img

        return img_aug

    def add_noise(self, img):
        np_img = np.array(img)
        noisy_img = random_noise(np_img, mode="gaussian", var=0.0009, clip=True)
        return PIL.Image.fromarray((noisy_img * 255).astype(np.uint8))

    @torch.no_grad()
    def remove_noise(self, img):
        self.model = self.model.cuda()
        img_t = functional.to_tensor(img).unsqueeze(0).cuda()
        img_r = self.model(img_t).cpu().squeeze(0).clamp(0, 1)

        return img_r
