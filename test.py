# import torch
#
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# from download import find_model
# from models import DiT_models
# from video_models import VideoDiT_models
# import argparse
#
# device = 'cuda'
# model = VideoDiT_models['DiT-XL/2'](
#     input_size=32,
#     num_classes=1000
# ).to(device)
# # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
# ckpt_path = '/home/bingliang/DiT/pretrained_models/DiT-XL-2-256x256.pt'
# state_dict = find_model(ckpt_path)
#
#
# def parse(d):
#     keys = list(d.keys())
#     domain = set([i.split('.')[0] for i in keys])
#     print(domain)
#
#
# def load_by_name(para_dict, name):
#     sub_dict = {}
#     for k, v in para_dict.items():
#         if k.split('.')[0] == name:
#             sub_dict[k[len(name) + 1:]] = v
#     return sub_dict
#
# def video_model_load_stage_dict_from_image(para_dict, video_model):
#     video_model.final_layer.load_state_dict(load_by_name(para_dict, 'final_layer'))
#     video_model.y_embedder.load_state_dict(load_by_name(para_dict, 'y_embedder'))
#     video_model.x_embedder.load_state_dict(load_by_name(para_dict, 'x_embedder'))
#     video_model.t_embedder.load_state_dict(load_by_name(para_dict, 't_embedder'))
#
#     video_model.final_layer.requires_grad_(False)
#     video_model.y_embedder.requires_grad_(False)
#     video_model.x_embedder.requires_grad_(False)
#     video_model.t_embedder.requires_grad_(False)
#
#     block_dict = load_by_name(para_dict, 'blocks')
#     for i, block in enumerate(video_model.blocks):
#         block.spacial_block.load_state_dict(load_by_name(block_dict, '{}'.format(i)))
#         block.spacial_block.requires_grad_(False)
#
#
# video_model_load_stage_dict_from_image(state_dict, model)
#
# torch.save(model.state_dict(), '/home/bingliang/DiT/pretrained_models/DiT-XL-2-256x256-video.pt')

from video_dataset import WebVid
from torchvision.utils import save_image
import torch
from diffusion import create_diffusion
from video_models import VideoDiT_models
from download import find_model
from diffusers.models import AutoencoderKL
import torch


def show_dataset(dataset):
    grid = [dataset[0][0], dataset[2][0], dataset[3][0], dataset[4][0]]
    grid = torch.cat(grid) / 2 + 0.5
    save_image(grid, 'video.png', nrow=8)


def load_model(device, ckpt_path='/home/bingliang/DiffusionArchitecture/pretrained_models/0010000.pt'):
    latent_size = 32
    model_type = 'DiT-XL/2'
    num_classes = 1000
    model = VideoDiT_models[model_type](
        input_size=latent_size,
        num_classes=num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    return model


def denoise_test(x, t, model, vae, noise_diffusion, sample_diffusion, device, fixed_noise=False, img_path='denoise.png'):
    """
        x: (L, C, H, W)
    """
    with torch.no_grad():
        # encode to latent space
        z = vae.encode(x).latent_dist.sample().mul_(0.18215)

        # forward process
        t = torch.tensor([t] * z.size(0), device=device)  # (L, )
        y = torch.tensor([1000] * z.size(0), device=device).long()  # (L, )
        noise = torch.randn_like(z).to(device)
        if fixed_noise:
            noise = noise[0:1] + torch.zeros_like(noise).to(device)
        z_t = noise_diffusion.q_sample(z, t, noise=noise)
        # classifier free guidance
        z_tc = torch.cat([z_t, z_t], 0)  # (2n, 4, W, W)
        y_null = torch.tensor([1000] * x.size(0), device=device).long()  # (n, ) Null token
        y = torch.cat([y, y_null], 0)  # (2n, )
        model_kwargs = dict(y=y, cfg_scale=4.0)

        # denoise process
        z_0 = sample_diffusion.p_sample_loop(
            model.forward_with_cfg, z_tc.shape, z_tc, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        z_0, _ = z_0.chunk(2, dim=0)  # Remove null class samples
        x_0 = vae.decode(z_0 / 0.18215).sample

        # inter-mediate feature
        x_t = vae.decode(z_t / 0.18215).sample
        grid = torch.cat([x, x_t, x_0], dim=0)
        save_image(grid, img_path, normalize=True, value_range=(-1, 1))


dataset = WebVid("/home/bingliang/data/WebVid2.5M/videos",
                 "/home/bingliang/data/WebVid2.5M/subset_new_info.json",
                 image_size=256, frame_size=16)
device = 'cuda'
model = load_model(device)
noise_diffusion = create_diffusion(timestep_respacing="")
sample_diffusion = create_diffusion(timestep_respacing="200")
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
x = dataset[0][0]



denoise_test(x.to(device), 0, model, vae, noise_diffusion, sample_diffusion, device, False, 'denoise.png')