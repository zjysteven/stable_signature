import warnings

from skimage import color
import torch
from torch import nn
from torchvision.transforms import functional

from .base import BaseAttack
from .model_utils import Colorizer

warnings.filterwarnings("ignore")


class Colorize(BaseAttack):
    """Apply colorization to the gray-scaled image."""

    def __init__(self):
        self._setup = False

    def setup(self, local_rank=None):
        self._setup = True
        self.colorizer = Colorizer(pretrained=True).eval().cuda()
        if local_rank is not None:  # DDP
            self.colorizer = torch.nn.parallel.DistributedDataParallel(
                self.colorizer, device_ids=[local_rank]
            )

    @torch.no_grad()
    def attack(self, x):
        if not self._setup:
            self.setup()
        assert (
            x.max() <= 1.0 and x.min() >= 0.0
        ), "Input tensor values should be in the range [0, 1]."

        img_ls = []
        for i, img in enumerate(x):
            img_rgb = functional.to_pil_image(img)
            img_lab = color.rgb2lab(img_rgb)
            img_l = img_lab[:, :, 0]
            tensor_l = torch.tensor(img_l, device=x.device, dtype=torch.float)[
                None, :, :
            ]
            img_ls.append(tensor_l)
        img_ls = torch.stack(img_ls, dim=0)

        out_abs = self.colorizer(img_ls)
        out_rgbs = []
        for i in range(img_ls.shape[0]):
            out_lab = torch.cat((img_ls[i], out_abs[i]), dim=0)
            out_rgb = color.lab2rgb(out_lab.cpu().numpy().transpose((1, 2, 0)))
            out_rgbs.append(torch.from_numpy(out_rgb.transpose((2, 0, 1))))
        return torch.stack(out_rgbs, dim=0).clamp(0, 1).to(x.device)
