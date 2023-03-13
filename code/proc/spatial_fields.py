from scipy import interpolate
from scipy import spatial
import pandas as pd
import numpy as np
import h5py


class TreeField(object):
    def __init__(self, tree, r_data, c_data):
        self.tree = tree
        self.r_data = r_data
        self.c_data = c_data

    def query(self, rs_rel, cs_rel, k = 20, dist_eps = 1e-2):
        """
        Arguments
        ---------
        - `rs_rel`, `cs_rel` --- Numpy arrays of same shape, with elements
            ranging from 0 to 1 that give the row and column indices of the
            query locations respectively.
        - `k` --- Integer. Number of nearest neighbors to interpolate between
        - `dist_eps` --- Distances less than `dist_eps` will be rounded up
            to `dist_eps`. This avoids division by zero and keeps the
            interpolated function smooth even when there are source points
            close together.
        Returns
        -------
        `r_data`, `c_data` --- Numpy arrays with same shape as `rs_rel`. 
            Result of the interpolatd query into the `r_data` and
            `c_data` objects of the Field object.
        """
        qlocs = np.stack([rs_rel, cs_rel], axis = -1)
        dists, ixs = self.tree.query(qlocs, k)
        dists[abs(dists) < dist_eps] = dist_eps
        weights = (1/dists)
        weights /= weights.sum(axis = -1, keepdims = True)

        if (isinstance(self.r_data, pd.Series)): r_data = self.r_data.values
        else: r_data = self.r_data
        if (isinstance(self.c_data, pd.Series)): c_data = self.c_data.values
        else: c_data = self.c_data

        r_data = (r_data[ixs] * weights).sum(axis = -1)
        c_data = (c_data[ixs] * weights).sum(axis = -1)
        return r_data, c_data

    def save(self, fname):
        with h5py.File(fname, 'w') as f:
            d = f.create_dataset('t_data', self.tree.data.shape, 'float64')
            d[...] = self.tree.data
            d = f.create_dataset('r_data', self.r_data.shape, 'float64')
            d[...] = self.r_data
            d = f.create_dataset('c_data', self.c_data.shape, 'float64')
            d[...] = self.c_data

    @staticmethod
    def load(fname):
        with h5py.File(fname, 'r') as dat:
            tree = spatial.KDTree(dat['t_data'][...])
            return Field(tree, dat['r_data'][...], dat['c_data'][...])




class LinearField(object):

    def __init__(self, r_grid, c_grid, r_data, c_data):
        self.r_grid = r_grid
        self.c_grid = c_grid
        self.r_data = r_data
        self.c_data = c_data
        eval_locs = np.stack([r_grid, c_grid], axis = -1).reshape(-1, 2)
        self.r_interp = interpolate.LinearNDInterpolator(
            eval_locs, r_data.ravel())
        self.c_interp = interpolate.LinearNDInterpolator(
            eval_locs, c_data.ravel())

    def query(self, rs_rel, cs_rel):
        qlocs = np.stack([rs_rel, cs_rel], axis = -1)
        r_data = self.r_interp(qlocs).reshape(rs_rel.shape)
        c_data = self.c_interp(qlocs).reshape(cs_rel.shape)
        return r_data, c_data

    def save(self, fname):
        with h5py.File(fname, 'w') as f:
            d = f.create_dataset('r_grid', self.r_grid.shape, 'float64')
            d[...] = self.r_grid
            d = f.create_dataset('c_grid', self.c_grid.shape, 'float64')
            d[...] = self.c_grid
            d = f.create_dataset('r_data', self.r_data.shape, 'float64')
            d[...] = self.r_data
            d = f.create_dataset('c_data', self.c_data.shape, 'float64')
            d[...] = self.c_data

    @staticmethod
    def load(fname):
        with h5py.File(fname, 'r') as dat:
            return LinearField(
                dat['r_grid'][...], dat['c_grid'][...],
                dat['r_data'][...], dat['c_data'][...])




