from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields
from proc import possible_fields
from proc import lsq_fields
from proc import video_gen
from proc import cornet

from proc import detection_task as det
import plot.behavioral as plot_bhv

from pprint import pprint
import numpy as np
import skvideo.io



if __name__ == '__main__':


    model, ckpt = cornet.load_cornet("Z")
    train_sizes = [2, 5, 10, 15, 20, 25, 30, 40]
    decoders = [(0, 4, 2)]

    isolated_task = det.IsolatedObjectDetectionTask('data/imagenet/index.csv',
        whitelist = ['n02808440', 'n07718747'])
    att_task = det.IsolatedObjectDetectionTask('data/imagenet/index.csv',
        whitelist = ['n02808440', 'n07718747'])


    # Training set size difficulty evaluation
    encodings = {}
    skregs = {}
    regmods = {}
    all_ys = {}
    for T in train_sizes:
        encodings[T], skregs[T], regmods[T], all_ys[T] = det.fit_logregs(
            model, decoders, isolated_task, train_size = T)
        det.save_logregs(regmods)
    

    ival_imgs, ival_ys = isolated_task.val_set(None, 40)
    aval_imgs, aval_ys = att_task.val_set(None, 40)
    encodings_val = det.model_encodings(model, decoders, ival_imgs)


