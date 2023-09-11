# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
import models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.image_size in [256, 512]

    # Load model:
    latent_size = args.image_size // 8
    model = models.VideoDiT_models[args.model](
        input_size=latent_size
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    origin_y = ['Merida, mexico - may 23, 2017: tourists are walking on a roadside near catholic church in the street of mexico at sunny summer day.',
        'Fun clown - 3d animation',
        '11th march 2017. nakhon pathom, thailand. devotees goes into a trance at the wai khru ceremony at wat bang phra temple. what bang phra is famous for its magically charged tattoos and amulets.',
        'Decorate with pineapple sweet cake roll.']
    y = []
    for text in origin_y:
        y += [text] * args.frame_size

    # Create sampling noise:
    n = len(y)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)  # (n, 4, W, W)

    # Setup classifier-free guidance:
    # TODO: add condition
    z = torch.cat([z, z], 0)  # (2n, 4, W, W)
    y_null = ["" for i in y]
    y = y + y_null
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save videos as image grid
    save_image(samples, "sample_video.png", nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(models.VideoDiT_models.keys()), default="FA-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--frame_size", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
