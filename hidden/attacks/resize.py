import numpy as np
from torchvision.transforms import functional

from .base import BaseAttack


class Resize(BaseAttack):
    """Perform resizing with a given scale

    Args:
        scale (float): target area scale (percentage of original image area)
    """

    def __init__(self, scale):
        assert scale > 0 and scale < 1, "Scale should be in (0, 1)"
        self.scale = np.sqrt(scale)
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        new_edges_size = [int(s * self.scale) for s in x.shape[-2:]]
        return functional.resize(x, new_edges_size)
