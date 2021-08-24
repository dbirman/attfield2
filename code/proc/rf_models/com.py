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
        keys `'com_x'`, `'com_y'`.
    """

    # Perform weighted average of row,col values by gradient intensity
    rs = np.arange(rf_grads.shape[-2])
    cs = np.arange(rf_grads.shape[-1])
    cs, rs = np.meshgrid(cs, rs)
    weight = rf_grads.sum(axis = -3)
    weight /= weight.sum(axis = (-2, -1), keepdims = True)
    r_com = (weight * rs[None, ...]).sum(axis = (-1, -2))
    c_com = (weight * cs[None, ...]).sum(axis = (-1, -2))
    return {'com_x': c_com, 'com_y': r_com}




