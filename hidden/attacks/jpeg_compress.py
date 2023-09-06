from augly.image import functional as aug_functional
import numpy as np
import torch
from torchvision.transforms import functional

from .base import BaseAttack


class JPEGCompress(BaseAttack):
    """Apply JPEG compression to an image given a quality factor

    Args:
        quality (int): JPEG encoding quality. 0 is lowest quality, 100 is highest.
    """

    def __init__(self, quality=50):
        assert (
            quality >= 0 and quality <= 100
        ), "JPEG quality should be between 0 and 100"
        self.quality = quality
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        img_aug = torch.zeros_like(x)
        for i, img in enumerate(x):
            pil_img = functional.to_pil_image(img)
            img_aug[i] = functional.to_tensor(
                aug_functional.encoding_quality(pil_img, quality=self.quality)
            )

        return img_aug
