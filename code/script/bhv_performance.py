import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/content/gdrive/My Drive/attfield/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

!pip3 -q install colormath scikit-video

import sys
for k in sys.modules:
    if "detection_task" in k:
        print(k)
        sys.modules.pop("proc.detection_task")

from proc import cornet
from proc import detection_task as det

import numpy as np
import tqdm

import warnings
warnings.filterwarnings("ignore")

# Set parameters
if __name__ == '__main__':
    TRAIN_NS = [2, 10]
    VAL_N = 10

    model, ckpt = cornet.load_cornet("Z")
    decoders = [(0, 4, 2)]
    isolated_task = det.IsolatedObjectDetectionTask(
        Paths.data.join('imagenet/index.csv'),
        whitelist = ['n02808440', 'n07718747'])
    att_task = det.FourWayObjectDetectionTask(
        Paths.data.join('imagenet/index.csv'),
        whitelist = ['n02808440', 'n07718747'])


    # Train models

    iso_train = {}
    att_train = {}

    for n in TRAIN_NS:
        iso_train_n = det.fit_logregs(model, decoders, isolated_task,
                                      train_size = n, shuffle = False)
        iso_train[n] = dict(zip(['enc', 'sk', 'nn', 'imgs', 'ys'], iso_train_n))
        att_train_n = det.fit_logregs(model, decoders, att_task,
                                      train_size = n, shuffle = False)
        att_train[n] = dict(zip(['enc', 'sk', 'nn', 'imgs', 'ys'], att_train_n))


    # Compare performance on train, validation, and crosstrained

    raw_preds = [
        'trained_on', 'train_n',
        'img_set', 'img_i', 'category',
        'gt', 'pred', 'score']
    raw_preds = dict(zip(raw_preds, [[] for _ in raw_preds]))
    scores_cat = [
        'trained_on', 'train_n',
        'img_set', 'category',
        'acc', 'auc', 'precision', 'recall',
        'total_acc', 'total_auc',
        'total_precision', 'total_recall'
    ]
    scores_cat = dict(zip(scores_cat, [[] for _ in scores_cat]))

    val_imgs = {}
    val_ys = {}
    val_imgs['iso'], val_ys['iso'] = isolated_task.val_set(
        None, VAL_N, shuffle = False)
    val_imgs['att'], val_ys['att'], val_locs = att_task.val_set(
        None, VAL_N, shuffle = False)
    val_enc = {
        k: det.model_encodings(model, decoders, val_imgs[k])
        for k in val_imgs}

    for n in TRAIN_NS:
        for task_name, trained in [('iso', iso_train),
                                   ('att', att_train)]:        
            
            # Iterate over validation set encodings
            # (potentially of the other task i.e. 'x')
            if task_name == 'iso': other_task = 'att'
            else: other_task = 'iso'
            val_enc_curr = val_enc[task_name]
            val_ys_curr = val_ys[task_name]
            x_val_enc_curr = val_enc[other_task]
            x_val_ys_curr = val_ys[other_task]
            for img_set, enc, curr_ys in [
                     ('train', trained[n]['enc'], trained[n]['ys']),
                     ('val', val_enc_curr, val_ys_curr),
                     ('x_val', x_val_enc_curr, x_val_ys_curr)]:
                            
                
                # Raw scores/predictions
                decision = det.multi_decision(trained[n]['nn'], enc)
                pred = det.by_cat(trained[n]['nn'],
                    lambda c: trained[n]['nn'][c].predict(enc[c]))
                for c in pred:
                    shp = pred[c].shape
                    raw_preds['score'].append(decision[c])
                    raw_preds['pred'].append(pred[c])
                    raw_preds['gt'].append(trained[n]['ys'][c])
                    raw_preds['category'].append(np.full(shp, c))
                    raw_preds['img_i'].append(np.arange(np.prod(shp)))
                    raw_preds['img_set'].append(np.full(shp, img_set))
                    raw_preds['train_n'].append(np.full(shp, n))
                    raw_preds['trained_on'].append(np.full(shp, task_name))
                                                   
                                                   
                # Performance Metrics
                _, overall, by_cat = det.score_logregs(
                    trained[n]['nn'], enc, curr_ys)
                for c in pred:
                    for k in ['recall', 'precision', 'auc', 'acc']:
                        scores_cat[k].append(by_cat[c][k])
                        scores_cat['total_'+k].append(overall[k])
                    scores_cat['precision'].append(by_cat[c]['precision'])
                    scores_cat['auc'].append(by_cat[c]['auc'])
                    scores_cat['acc'].append(by_cat[c]['acc'])
                    scores_cat['category'].append(c)
                    scores_cat['img_set'].append(img_set)
                    scores_cat['train_n'].append(n)
                    scores_cat['trained_on'].append(task_name)