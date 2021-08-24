from proc import detection_task as det

def generate_set(img, n, wl = None):
    """
    Group metadata includes:
    - `ys` --- Label 1 or 0 indicating target presence or absence,
        shape: (n,) corresponding for first dim of images array.
    Arguments
    ---------
    - `img` --- Path to HDF5 archive of distilled detection dask.
    - `n` --- Number of images per category.
    - `wl` --- Whitelist of category names to include. If None, all
        categories will be used.
    """
    four_task = det.DistilledDetectionTask(
        img, whitelist = wl)
    imgs, ys, _, _ = four_task.val_set(None, n)
    group_info = [
        (c, {'ys': ys[c]})
        for c in imgs]
    return group_info, lambda: imgs.values() 
