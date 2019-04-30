from proc import network_manager as nm
from proc import video_gen

from sklearn.linear_model import LogisticRegression
import scipy.ndimage.interpolation as scimg
from skimage.transform import rescale
import sklearn.metrics as sk_metric
from torch import nn
import pandas as pd
import numpy as np
import skvideo.io
import operator
import torch
import tqdm
import os



class LayerBypass(nm.LayerMod):

    def pre_layer(self, *args, **kwargs):
        return args, kwargs, args[0]
    def post_layer(self, outputs, cache):
        return cache


class ClassifierMod(nm.LayerMod):
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def post_layer(self, outputs, cache):
        return self.decision_function(outputs)

    def predict(self, encodings):
        return self.decision_function(encodings) > 0

    def decision_function(self, encodings):
        if isinstance(encodings, np.ndarray):
            return np.dot(encodings, self.w.T[:, 0]) + self.b
        else:
            w = torch.tensor(self.w.T).float()
            b = torch.tensor(self.b).float()
            return torch.mm(encodings, w) + b

    def predict_on_fn(self, decision_fn):
        return decision_fn > 0


def load_logregs(filename):
    archive = np.load(filename)
    if archive['type'] == 'mod':
        return {
            c: ClassifierMod(archive[c+'_w'], archive[c+'_b'])
            for c in archive['categories']
        }
    else:
        regs = {c: LogisticRegression()
                for c in archive['categories']}
        for c in regs:
            regs[c].coef_ = archive[c+'_c']
            regs[c].intercept_ = archive[c+'_i']



def save_logregs(filename, logregs):
    logreg_type = type(logregs[next(iter(logregs.keys()))])
    if logreg_type is ClassifierMod:
        np.savez(filename,
            categories = list(logregs.keys()),
            **{c+"_w": logregs[c].w for c in logregs},
            **{c+"_b": logregs[c].b for c in logregs},
            type = 'mod')
    else:
        np.savez(filename,
            categories = list(logregs.keys()),
            **{c+"_c": logregs[c].coef_ for c in logregs},
            **{c+"_i": logregs[c].intercept_ for c in logregs},
            type = 'sk')


def multi_decision(regs, encodings):
    '''
    Utility function for running the decision function of each
    category regression'''
    return {c: regs[c].decision_function(encodings[c])
            for c in regs}

def by_cat(guide_dict, func):
    '''
    Utility function for running functions across categories 
    of a task.'''
    return {c: func(c) for c in guide_dict}


class IsolatedObjectDetectionTask():
    def __init__(self, index, image_size = 224,
        whitelist = None):
        '''
        ### Arguments
        - `index` --- Path to an index.csv file
        '''
        self.base_dir = os.path.dirname(index)
        self.index = pd.read_csv(index)

        if whitelist is not None:
            row_whitelist = np.isin(self.index['wnid'], whitelist)
            self.index = self.index.loc[row_whitelist]

        self.cats = self.index['category'].unique()
        self.image_size = 224


    def train_set(self, cat, n, **kwargs):
        '''
        Pull images from ILSVRC2014 validation set of a given
        category. These are from the validation set so that
        earlier layers of the model will not have been trained
        on them.
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        ### Returns
        - `imgs` --- A collection of training images for a
            binary classification of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''
        if cat is None:
            # Run _load_set for each category and arrange the
            # returned values into dictionaries
            rets = None
            for c in self.cats:
                curr_ret = self._load_set(c, n, 'n', **kwargs)
                if rets is None:
                    rets = tuple({} for i in range(len(curr_ret)))
                for i in range(len(curr_ret)):
                    rets[i][c] = curr_ret[i]
            return rets
        else:
            return self._load_set(cat, n, 'n', **kwargs)


    def _crop(self, img):
        '''Zoom and center crop to self.image_size'''
        factor = self.image_size/min(img.shape[1], img.shape[2])
        img = scimg.zoom(img, [1, factor, factor, 1], order = 1)
        #rescale(img[0], factor, factor, anti_aliasing = True)

        # Center crop
        # See https://stackoverflow.com/a/50322574/1888160
        bounding = (1, self.image_size, self.image_size, 3)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape,
                          bounding))
        end = tuple(map(operator.add, start, bounding))
        slices = tuple(map(slice, start, end))
        return img[slices]


    def _load_images(self, idxs):
        '''Convert row indexes in self.index to zoommed and cropped
        images as numpy arrays.
        ### Arguments
        - `idxs` --- List of integers indexing rows of self.index
        ### Returns
        - `imgs` --- Numpy array containing images ordered as `idxs`.
            Will be of shape:
            (len(idxs), self.image_size, self.image_size, 3)'''
        paths = self.index['path'].iloc[idxs]
        return np.concatenate([
            self._crop(skvideo.io.vread(os.path.join(self.base_dir, pth)))
            for pth in paths])


    def _load_set(self, cat, n, ncol, cache = 'data/.imagenet_cache',
                  shuffle = False):
        '''
        Pull positive and negative examples of a given category
        These will be the first `n` images of the category, as ordered
        by ncol
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        - `ncol` --- The column from which to pull an ordering over
            the images
        ### Returns
        - `imgs` --- A collection of images for a binary
            classification. Will be of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''

        if cache is not None:
            cachefile = os.path.join(cache, cat + str(n) + ncol + 
                                     str(self.image_size) + ".npz")

        if cache is not None and os.path.exists(cachefile):
            archive = np.load(cachefile)
            imgs = archive['imgs']
            ys = archive['ys']

        else:

            # Identify the first `n` images in this category
            cat_images = self.index['category'] == cat
            first_n = self.index[ncol] < n
            true_idxs = np.where(cat_images & first_n)[0]
            true_images = self._load_images(true_idxs)

            # Get the first images spread across false categories
            false_idxs = np.where(~cat_images & first_n)[0]
            n_sort = np.argsort(self.index[ncol].iloc[false_idxs])
            false_idxs = false_idxs[n_sort[:n]]
            false_images = self._load_images(false_idxs)

            imgs = np.concatenate([true_images, false_images])
            ys = np.concatenate([np.ones(n, dtype = bool),
                                 np.zeros(n, dtype = bool)])

            if cache is not None:
                np.savez(cachefile, imgs = imgs, ys = ys)

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(2*n)
            ys = ys[shuf]
            imgs = imgs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
        imgs = video_gen.batch_transform(imgs, video_gen.normalize)
        return imgs, ys


    def val_size(self, train_size):
        max_ns = [
            max(self.index.where(self.index['category'] == c)['n'])
            for c in self.cats]
        return min(max_ns) - train_size


    def val_set(self, cat, n, **kwargs):
        '''See documentation for train_set'''
        if cat is None:
            # Run _load_set for each category and arrange the
            # returned values into dictionaries
            rets = None
            for c in self.cats:
                curr_ret = self._load_set(c, n, 'neg_n', **kwargs)
                if rets is None:
                    rets = tuple({} for i in range(len(curr_ret)))
                for i in range(len(curr_ret)):
                    rets[i][c] = curr_ret[i]
            return rets
        else:
            return self._load_set(cat, n, 'neg_n', **kwargs)



class FourWayObjectDetectionTask(IsolatedObjectDetectionTask):

    def _load_set(self, cat, n, ncol, cache = 'data/.imagenet_cache',
        loc = -1, shuffle = False):
        '''
        Pull positive and negative examples of a given category
        These will be the first `n` images of the category, as ordered
        by ncol
        ### Arguments
        - `cat` --- A category identifier from the self.cats
        - `n` --- Number of positive images to include
        - `ncol` --- The column from which to pull an ordering over
            the images
        - `loc` --- Position where the first image from `imgs` should
            go in the grid. The mapping is
            - 0: Top Left
            - 1: Top Right
            - 2: Bottom Left
            - 3: Bottom Right
            - -1: Random
        - `shuffle` --- If true, `imgs` will have all the positive 
            examples grouped together in the beginning of the first
            dimension. If false then `imgs` will be shuffled along 
            the first dimension.
        ### Returns
        - `imgs` --- A collection of images for a binary
            classification. Will be of size (2*n, 3, 224, 224).
            These will contain `n` images containing the object
            category and `n` not containing the object category
            in random order.
        - 'ys' --- An array of shape (2*n,) taking values either
            1 or 0 to indicate whether each image in `imgs`
            contains an object of the category
        '''

        if cache is not None:
            cachefile = os.path.join(cache, "4W" + str(loc) + cat +  
                     str(n) + ncol + str(self.image_size) + ".npz")

        if cache is not None and os.path.exists(cachefile):
            archive = np.load(cachefile)
            imgs = archive['imgs']
            ys = archive['ys']
            locs = archive['locs']

        else:
            # Identify the first `n` images in this category
            cat_images = self.index['category'] == cat
            first_n = self.index[ncol] < n
            true_idxs = np.where(cat_images & first_n)[0]
            true_images = self._load_images(true_idxs)

            # Get the first images spread across false categories
            # We need 7 times as many images for false as true so
            # we can both generate the negative examples and fill
            # in the three other spaces in positive examples
            first_false_n = self.index[ncol] < 7*n/(len(self.cats)-1)
            false_idxs = np.where(~cat_images & first_false_n)[0]
            n_sort = np.argsort(self.index[ncol].iloc[false_idxs])
            false_idxs_posarr = false_idxs[n_sort[:3*n]]
            false_images_posarr = self._load_images(false_idxs_posarr)
            
            false_idxs = false_idxs[n_sort[3*n:7*n]]
            false_images = self._load_images(false_idxs)

            pos_imgs = []
            neg_imgs = []
            pos_locs = []
            neg_locs = []
            for i in range(n):
                array_posimg = false_images_posarr[3*i:3*(i+1)]
                pos_img, pos_loc = self._arrange_grid(
                    np.concatenate([[true_images[i]], array_posimg]),
                    loc = loc)
                neg_img, neg_loc = self._arrange_grid(
                    false_images[4*i:4*(i+1)],
                    loc = loc)
                pos_imgs.append(pos_img)
                neg_imgs.append(neg_img)
                pos_locs.append(pos_loc)
                neg_locs.append(neg_loc)

            imgs = np.concatenate([pos_imgs, neg_imgs])
            ys = np.concatenate([np.ones(n, dtype = bool),
                                 np.zeros(n, dtype = bool)])
            locs = np.concatenate([pos_locs, neg_locs])

            if cache is not None:
                np.savez(cachefile, imgs = imgs, ys = ys, locs = locs)

        # Shuffle
        if shuffle:
            shuf = np.random.permutation(2*n)
            ys = ys[shuf]
            imgs = imgs[shuf]
            locs = locs[shuf]

        # Normalize how CORNet expects, and shift channel
        # dimension to torch format
        imgs = torch.tensor(np.moveaxis(imgs, -1, 1)).float()
        imgs = video_gen.batch_transform(imgs, video_gen.normalize)
        return imgs, ys, locs


    def _arrange_grid(self, imgs, loc = -1):
        '''
        Arrange four images into 2x2 grid, with the first image at a
        specified location and others distributed randomly.
        ### Arguments
        - `imgs` --- A list or array of four images, each with shape
            (h, w, channel) to be arranged into a 2x2 grid.
        - `loc` --- Position where the first image from `imgs` should
            go in the grid. The mapping is
            - 0: Top Left
            - 1: Top Right
            - 2: Bottom Left
            - 3: Bottom Right
            - -1: Random
        ### Returns
        - `grid` --- The image grid, of shape (2*h, 2*w, channel)
        - `loc` --- The passed parameter for `loc`, or the equivalent
            mapping of what was randomly assigned otherwise.
        '''
        # Rearrange `imgs` to put the first image in the right spot
        # and to randomize position of the other images
        
        if loc == -1:
            loc = np.random.randint(4)

        rand = np.array(imgs)[np.random.permutation(3)+1]
        if loc == 0:
            imgs = [imgs[0], rand[0], rand[1], rand[2]]
        elif loc == 1:
            imgs = [rand[0], imgs[0], rand[1], rand[2]]
        elif loc == 2:
            imgs = [rand[0], rand[1], imgs[0], rand[2]]
        elif loc == 3:
            imgs = [rand[0], rand[1], rand[2], imgs[0]]

        return np.concatenate([
                np.concatenate([imgs[0], imgs[1]], axis = 1),
                np.concatenate([imgs[2], imgs[3]], axis = 1)],
            axis = 0), loc


def score_logregs(logregs, encodings, ys):

    per_cat = {}
    preds = {}
    decision = {}
    for c in encodings:

        # Get predictions and decision function values
        preds[c] = logregs[c].predict(encodings[c])
        decision[c] = logregs[c].decision_function(encodings[c])

        per_cat[c] = {
            'acc': sk_metric.accuracy_score(ys[c], preds[c]),
            'precision': sk_metric.precision_score(ys[c], preds[c]),
            'recall': sk_metric.recall_score(ys[c], preds[c]),
            'auc': sk_metric.roc_auc_score(ys[c], decision[c])
        }

    ys_concat = np.concatenate([ys[c] for c in encodings])
    preds_concat = np.concatenate([preds[c] for c in encodings])
    decision_concat = np.concatenate([decision[c] for c in encodings])
    overall = {
        'acc': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'precision': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'recall': sk_metric.accuracy_score(
            ys_concat, preds_concat),
        'auc': sk_metric.roc_auc_score(
            ys_concat, decision_concat)
    }

    return preds, overall, per_cat




def model_encodings(model, decoders, imgs, mods = {}):
    '''
    ### Arguments
    - `model` --- A PyTorch model, presumably CORNet
    - `decoders` --- A list of layer indices (tuples of ints)
        giving layers that perform 'decoding' functions in
        the model and so should be skipped.
    - `imgs` --- Either an array of shape (n, h, w, c) giving
        the images to be encoded, or a dictionary mapping
        categories to arrays of that type.
    ### Returns
    - `encodings` --- Model encodings of `imgs`. Either an array
        of shape (n_imgs, n_features) if `imgs` was an array, or
        a dictionary mapping categories to such arrays of `imgs`
        was passed as a dictionary.
    '''
    if isinstance(imgs, dict):
        return {c: model_encodings(model, decoders, imgs[c])
                for c in imgs}
    else:
        decoder_bypass = {l: LayerBypass() for l in decoders}
        mgr = nm.NetworkManager.assemble(model, imgs,
                mods = {**mods, **decoder_bypass})
        return mgr.computed[(0,)].detach().numpy()



def fit_logregs(model, decoders, task, mods = {},
                train_size = 30, shuffle = False):
    '''
    Fit linear decoders to a model and a task.
    ### Arguments
    - `model` --- The neural network that will be performing the task
    - `decoders` --- A list of layer indexes (tuples of ints) that
        perform decoding or behavioral task operations. These will be 
        skipped when computing the model on task inputs and be replaced
        by a task-specific logistic regression decoder for each class.
        It is expected that the model will give output in a flat format
        with shape (batch, features). If this is not the case once
        decoders are bypassed then a LayerMod should be included on 
        layer (0,) that will flatten the output.
    - `mods` --- Any LayerMods to be applied to `model`, for example
        attention mechanisms could be passed here if decoders need to
        adapt to them.
    '''

    # Compute how the network embeds the images in feature space
    encodings = {}
    all_ys = {}
    all_imgs = {}
    skregs = {}
    regmods = {}

    for c in task.cats:
        print("Training class:", c)
        imgs, ys = task.train_set(c, train_size, shuffle=shuffle)[:2]
        all_ys[c] = ys
        all_imgs[c] = imgs
        encodings[c] = model_encodings(
            model, decoders, imgs, mods = mods)

        skregs[c] = LogisticRegression().fit(encodings[c], ys)
        regmods[c] = ClassifierMod(
            skregs[c].coef_,
            skregs[c].intercept_)

    return encodings, skregs, regmods, all_imgs, all_ys



    



