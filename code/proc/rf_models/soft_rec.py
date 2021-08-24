import numpy as np

def summarize(rf_grads):
    """Map a receptive field heatmap gradient to per-unit center of mass

    Arguments
    ---------
    - `rf_grads` --- Numpy array, shape (n_unit, channel, row, col)
        Not assumed to have any particular sign. Cancellation may
        occur if absolute gradients are not given.

    Returns
    -------
    - `params` --- Dict of numpy arrays, each shape (n_unit,), with
        keys `'com_x'`, `'com_y'`, `x0`, `y0`, `x1`, `y1`, `w`, `h`, `a`, `rad`.
    """

    # Perform weighted average of row,col values by gradient intensity
    rs = np.arange(rf_grads.shape[-2])
    cs = np.arange(rf_grads.shape[-1])
    cs, rs = np.meshgrid(cs, rs)
    tot_grad = rf_grads.sum(axis = -3)
    weight = tot_grad.sum(axis = (-2, -1), keepdims = True)
    r_com = (weight * rs[None, ...]).sum(axis = (-1, -2))
    c_com = (weight * cs[None, ...]).sum(axis = (-1, -2))

    thresh = tot_grad.max(axis = (-1, -2)) * 0.2
    masks = [np.where(arr >= thresh[i]) for i, arr in enumerate(tot_grad)]
    x0 = np.array([np.min(m[1]) for m in masks])
    y0 = np.array([np.min(m[0]) for m in masks])
    x1 = np.array([np.max(m[1]) for m in masks])
    y1 = np.array([np.max(m[0]) for m in masks])
    w = x1 - x0
    h = y1 - y0
    a = w * h
    rad = (w + h) / 2

    return {
        'com_x': c_com, 'com_y': r_com,
        'x0':x0,
        'x1':x1,
        'y0':y0,
        'y1':y1,
        'w':w,
        'h':h,
        'a':a,
        'rad':rad,
    }




