import numpy as np
from torchvision.transforms import RandomResizedCrop

from .base import BaseAttack


class ResizedCrop(BaseAttack):
    """Random crop 25% the original image area and resize to the original size

    Args:
        img_size (int): size of the original image
        ratio (float): lower and upper bounds for the random aspect ratio of the crop, before resizing
    """

    def __init__(self, img_size=512, scale=(0.5, 0.5), ratio=(0.75, 1.3333333333333333)):
        self.transform = RandomResizedCrop(
            img_size,
            scale=scale,
            ratio=ratio
        )
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        return self.transform(x)
