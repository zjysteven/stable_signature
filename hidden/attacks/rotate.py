import numpy as np
from torchvision.transforms import functional

from .base import BaseAttack


class Rotate(BaseAttack):
    """Perform rotation with a given angle

    Args:
        angle (float): angle in degrees
    """

    def __init__(self, angle=None):
        self.angle = angle
        self.setup()

    def setup(self, **kwargs):
        pass

    def attack(self, x):
        if self.angle is None:
            angle = int(np.random.choice(range(-90, 90)))
        else:
            angle = self.angle
        return functional.rotate(x, angle)
