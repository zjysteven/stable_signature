import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional

from .base import BaseAttack


class ColorJitter(BaseAttack):
    """Adjust brightness with a given factor

    Args:
        factor (float): non-negative value, where 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    """

    def __init__(self):
        self.transform = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
        )
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        img_aug = torch.zeros_like(x)
        for i, img in enumerate(x):
            pil_img = functional.to_pil_image(img)
            img_aug[i] = functional.to_tensor(
                self.transform(pil_img)
            )
        return img_aug
