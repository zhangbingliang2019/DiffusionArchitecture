import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from download import find_model
from models import DiT_models
from video_models import VideoDiT_models
import argparse

device = 'cuda'
model = VideoDiT_models['DiT-XL/2'](
    input_size=32,
    num_classes=1000
).to(device)
# Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
ckpt_path = '/home/bingliang/DiT/pretrained_models/DiT-XL-2-256x256.pt'
state_dict = find_model(ckpt_path)


def parse(d):
    keys = list(d.keys())
    domain = set([i.split('.')[0] for i in keys])
    print(domain)


def load_by_name(para_dict, name):
    sub_dict = {}
    for k, v in para_dict.items():
        if k.split('.')[0] == name:
            sub_dict[k[len(name) + 1:]] = v
    return sub_dict

def video_model_load_stage_dict_from_image(para_dict, video_model):
    video_model.final_layer.load_state_dict(load_by_name(para_dict, 'final_layer'))
    video_model.y_embedder.load_state_dict(load_by_name(para_dict, 'y_embedder'))
    video_model.x_embedder.load_state_dict(load_by_name(para_dict, 'x_embedder'))
    video_model.t_embedder.load_state_dict(load_by_name(para_dict, 't_embedder'))

    video_model.final_layer.requires_grad_(False)
    video_model.y_embedder.requires_grad_(False)
    video_model.x_embedder.requires_grad_(False)
    video_model.t_embedder.requires_grad_(False)

    block_dict = load_by_name(para_dict, 'blocks')
    for i, block in enumerate(video_model.blocks):
        block.spacial_block.load_state_dict(load_by_name(block_dict, '{}'.format(i)))
        block.spacial_block.requires_grad_(False)


video_model_load_stage_dict_from_image(state_dict, model)

torch.save(model.state_dict(), '/home/bingliang/DiT/pretrained_models/DiT-XL-2-256x256-video.pt')