from compressai.zoo import cheng2020_anchor
import torch
from torchvision.transforms import functional

from .base import BaseAttack


class CompressyAutoEncode(BaseAttack):
    def __init__(self, quality=6):
        self._setup = False
        self.quality = quality

    def setup(self, local_rank=None):
        self._setup = True
        self.net = (
            cheng2020_anchor(self.quality, metric="mse", pretrained=True).eval().cuda()
        )
        if local_rank is not None: # DDP
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net, device_ids=[local_rank]
            )

    @torch.no_grad()
    def attack(self, x):
        if not self._setup:
            self.setup()
        out = self.net.forward(x)
        return out['x_hat'].clamp(0, 1)
