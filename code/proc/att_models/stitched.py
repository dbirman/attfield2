import torch
from proc import network_manager as nm

def attn_model(gain_split, splits, merges, beta):
    return nm.mod_merge(
        {tuple(gain_split): Cutter(beta)},
        {tuple(s): Cutter(1.) for s in splits},
        {tuple(m): Stitcher() for m in merges},
    )

class Cutter(nm.LayerMod):
    def __init__(self, beta):
        super(Cutter, self).__init__()
        self.beta = beta

    def pre_layer(self, inp, *args, **kwargs):
        if inp.shape[-1] % 2 != 0 or inp.shape[-2] % 2 != 0:
            raise ValueError("Input must have even shape.")
        nr, nc = inp.shape[-2:]
        top_lf = inp[..., :nr//2, :nc//2]
        top_rt = inp[..., :nr//2, nc//2:]
        bot_lf = inp[..., nr//2:, :nc//2]
        bot_rt = inp[..., nr//2:, nc//2:]
        top_lf = top_lf * self.beta
        out = torch.cat([top_lf, top_rt, bot_lf, bot_rt], dim=0)
        return (out,) + args, kwargs, None

class Stitcher(nm.LayerMod):
    def post_layer(self, output, chache):
        if output.shape[0] % 4 != 0:
            raise ValueError("Input must have batch dim be a multiple of 4.")
        n = output.shape[0]
        top_lf = output[0*(n//4):1*(n//4)]
        top_rt = output[1*(n//4):2*(n//4)]
        bot_lf = output[2*(n//4):3*(n//4)]
        bot_rt = output[3*(n//4):]
        out = torch.cat([
            torch.cat([top_lf, top_rt], dim = -1),
            torch.cat([bot_lf, bot_rt], dim = -1)
        ], dim=-2)
        return out








