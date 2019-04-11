from proc import attention_models as att
from proc import network_manager as nm
from proc import lsq_fields
from proc import cornet

import pandas as pd
import numpy as np



if __name__ == '__main__':

    model, ckpt = cornet.load_cornet("Z")
    layers = [(0, 1)]
    voxels, _, refined = lsq_fields.lsq_rfs(model, layers, verbose = 1,
        bar_speed = 12., percent = 2,
        grad_n = 10, grad_mode = 'slider')
    _, _, refined_att = lsq_fields.lsq_rfs(model, layers, verbose = 1,
        bar_speed = 12., percent = 2,
        grad_n = 10, grad_mode = 'slider',
        voxels = voxels,
        mods = {(0,): att.GaussianSpatialGain((30, 30), 20, 1.5)})

    lsq_fields.save_rf_csv("data/sgrf_nomod.csv", voxels, refined)
    lsq_fields.save_rf_csv("data/sgrf_withmod.csv", voxels, refined_att)