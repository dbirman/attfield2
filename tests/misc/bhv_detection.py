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

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib import colors
import matplotlib.patches as patches
import seaborn as sns
sns.set(color_codes = True)
sns.set_style('ticks')


if __name__ == '__main__':

    SIZE = 30

    model, ckpt = cornet.load_cornet("Z")
    task = det.IsolatedObjectDetectionTask('data/imagenet/index.csv',
        whitelist = ['n02808440', 'n07718747'])
    encodings, skregs, regmods, all_ys = det.fit_logregs(
        model, [(0, 4, 2)], task, train_size = SIZE)

    imgs_val, ys_val = task.val_set(None, SIZE)
    encodings_val = det.model_encodings(model, [(0, 4, 2)], imgs_val)

    _, scores_sk, _ = det.score_logregs(skregs, encodings, all_ys)
    _, scores_nn, _ = det.score_logregs(regmods, encodings, all_ys)

    print("SCIKIT:")
    pprint(scores_sk)

    print("TORCH:")
    pprint(scores_nn)


    task_4 = det.FourWayObjectDetectionTask('data/imagenet/index.csv',
        whitelist = ['n02808440', 'n07718747'], image_size = 112)
    imgs_4, ys_4, _ = task_4.val_set(None, SIZE)
    encodings_4 = det.model_encodings(model, [(0, 4, 2)], imgs_4)

    encodings_4t, skregs_4, regmods_4, ys_4t = det.fit_logregs(
        model, [(0, 4, 2)], task_4, train_size = SIZE)

    print("Four Way Task (Train):")
    _, scores_4t, _ = det.score_logregs(regmods_4, encodings_4t, ys_4t)
    pprint(scores_4t)

    print("Four Way Task (Crosstrained):")
    _, scores_4, _ = det.score_logregs(regmods, encodings_4, ys_4)
    pprint(scores_4)



    # Look at the decision landscape
    full_dec = [det.multi_decision(regmods, encodings),
                det.multi_decision(regmods, encodings_val),
                det.multi_decision(regmods, encodings_4),
                det.multi_decision(regmods_4, encodings_4t)]
    full_y = [det.by_cat(all_ys, lambda c: all_ys[c]),
              det.by_cat(all_ys, lambda c: ys_val[c]),
              det.by_cat(all_ys, lambda c: ys_4[c]),
              det.by_cat(all_ys, lambda c: ys_4t[c])]
    plot_bhv.task_performance(
        'plots/test/decision_fn_{}.pdf'.format(SIZE),
        full_dec, full_y,
        ["Train", "Validation", "FourWayX", "FourWay"])