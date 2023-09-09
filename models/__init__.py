import models.fully_attention as fa
import models.spacial_temporal_attention as sta

VideoDiT_models = {
    'FA-XL/2': fa.DiT_XL_2, 'FA-L/4': fa.DiT_L_4, 'FA-B/4': fa.DiT_B_4,
    'STA-XL/2': sta.DiT_XL_2, 'STA-L/4': sta.DiT_L_4, 'STA-B/4': sta.DiT_B_4
}


def test():
    import torch
    model = VideoDiT_models['STA-L/4']().to('cuda')
    x = torch.randn(32, 4, 32, 32).cuda()
    y = ['a boy'] * 2
    t = torch.randint(0, 10, size=(2,)).cuda()
    return model.forward(x, t, y)
