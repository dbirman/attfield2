import h5py
import torch

def generate_set(img, n, bl = None):
    """
    Group metadata includes:
    - `ys` --- Label 1 or 0 indicating target presence or absence,
        shape: (n,) corresponding for first dim of images array.
    Arguments
    ---------
    - `img` --- Path to HDF5 archive of images with a dataset for
      each category and `category_y` containing present/absen
      ground truth for the class.
    - `n` --- Number of images per category.
    - `bl` --- Category blacklist
    """
    img_h5 = h5py.File(img, 'r') 
    cats = [c for c in img_h5.keys() if not (c.endswith("_y") or c.endswith('.meta'))]
    if bl is not None:
        cats = [c for c in cats if c not in bl]
    group_info = [
        (c, {'ys': img_h5[c + '_y'][:n]})
        for c in cats]
    return group_info, lambda: (
        torch.Tensor(img_h5[c][:n]).float().permute(0, 3, 1, 2)
        for c in cats)