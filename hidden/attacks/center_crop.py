import numpy as np
from torchvision.transforms import functional

from .base import BaseAttack


class CenterCrop(BaseAttack):
    """Perform center crop such that the target area of the crop is at a given scale

    Args:
        scale (float): target area scale (percentage of original image area)
    """

    def __init__(self, scale=None):
        if scale is not None:
            assert scale > 0 and scale < 1, "Scale should be in (0, 1)"
            self.scale = np.sqrt(scale)
        else:
            self.scale = None
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        if self.scale is None:
            scale = np.sqrt(np.random.uniform(0.1, 1))
            new_edges_size = [int(s * scale) for s in x.shape[-2:]]
        else:
            new_edges_size = [int(s * self.scale) for s in x.shape[-2:]]
        return functional.center_crop(x, new_edges_size)
