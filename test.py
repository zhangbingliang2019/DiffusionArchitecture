# from models import DiT
# from video_models import get_1d_sincos_pos_embed, VideoDiT
# import torch
#
# model = VideoDiT(32, 16, 2, 1, 4, 1152,
#             28, 16, 4, 0.1, 1000)
#
# # (B, C, H, W)
# x = torch.randn(2 * 16, 4, 32, 32)
# t = torch.tensor([10]*16 + [11]*16)
# y = torch.tensor([2]*16 + [3]*16)
#
# model.forward(x, t, y)
# # frame_embed = get_1d_sincos_pos_embed(512, 16)
# #
# # print(frame_embed)

from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


data_path = '/home/bingliang/data/ImageNet/ILSVRC/Data/CLS-LOC/val'
transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
dataset = ImageFolder(data_path, transform=transform)
print(dataset)