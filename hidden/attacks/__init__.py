from .auto_encode import CompressyAutoEncode
from .base import BaseAttack
from .center_crop import CenterCrop
from .color_jitter import ColorJitter
from .colorize import Colorize
from .comb import CombAttack
from .guassian_denoise import GaussianDenoise
from .jpeg_compress import JPEGCompress
from .motion_deblur import MotionDeblur
from .resize import Resize
from .resized_crop import ResizedCrop
from .rotate import Rotate

# ----------------- attack options ----------------- #
attack_modules = {
    # ------------- conventional ------------- #
    "center_crop_0.3": CenterCrop(0.3),
    "color_jitter": ColorJitter(),
    "jpeg_compress_50": JPEGCompress(quality=50),
    "resized_crop": ResizedCrop(),
    "resize_0.3": Resize(0.3),
    "resize_0.7": Resize(0.7),
    "rotate": Rotate(),
    # ------------- ours ------------- #
    "colorize": Colorize(),
    "motion_deblur_5": MotionDeblur(kernel_size=(5,5), remove_blur=True),
    "motion_deblur_11": MotionDeblur(kernel_size=(11,11), remove_blur=True),
    "motion_deblur_17": MotionDeblur(kernel_size=(17,17), remove_blur=True),
    "gaussian_denoise": GaussianDenoise(),
    "auto_encode_6": CompressyAutoEncode(quality=6),
    "auto_encode_4": CompressyAutoEncode(quality=4),
    "auto_encode_2": CompressyAutoEncode(quality=2),
}
