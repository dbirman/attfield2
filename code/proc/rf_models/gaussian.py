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
        keys
        - `'com_x'`
        - `'com_y'`
        - `'var_x'`
        - `'var_y'`
        - `'cov'`
        - `'major_x'`
        - `'major_y'`
        - `'major_sigma'`
        - `'minor_x'`
        - `'minor_y'`
        - `'minor_sigma'`
    """

    # Perform weighted average of row,col values by gradient intensity
    rs = np.arange(rf_grads.shape[-2])
    cs = np.arange(rf_grads.shape[-1])
    cs, rs = np.meshgrid(cs, rs)
    weight = rf_grads.sum(axis = -3)
    weight /= weight.sum(axis = (-2, -1), keepdims = True)
    r_com = (weight * rs[None, ...]).sum(axis = (-1, -2))
    c_com = (weight * cs[None, ...]).sum(axis = (-1, -2))

    # Compute covariance
    cov = np.array([
        np.cov(np.stack([rs.ravel(), cs.ravel()]), aweights = warr.ravel())
        for warr in weight
    ])

    # Compute directional SDs
    eigh, eigv = zip(*[np.linalg.eig(cov[i]) for i in range(len(cov))])
    eigh = np.array(eigh)
    eigv = np.array(eigv)


    return {
        'com_x': c_com, 
        'com_y': r_com,
        'var_x': cov[:, 1, 1],
        'var_y': cov[:, 0, 0],
        'cov': cov[:, 1, 0],
        'major_x': eigv[:, 1, 0],
        'major_y': eigv[:, 0, 0],
        'major_sigma': abs(eigh[:, 0]),
        'minor_x': eigv[:, 1, 1],
        'minor_y': eigv[:, 0, 1],
        'minor_sigma': abs(eigh[:, 1]),
    }




