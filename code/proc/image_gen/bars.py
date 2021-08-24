import numpy as np
import torch

def generate_set(ns, size):
    """
    Three groups:
    - `grd` --- Grid of dots
    - `hor` --- Horizonatal bars
    - `ver` --- Vertical bars
    - `lat` --- Both horizonal and vertical bars.
    Group metadata:
    - `ns` --- Approximate number of bars in each image, used to 
        calculate the stride value, `size//n`.

    ##### Arguments
    - `ns` --- List of ints, each specifying the number of bars to
        give in an input image.
    - `size` --- Size of (square) input images.
    """
    def group_iter():
        # Gridded dots
        bars = torch.zeros((len(ns), 1, size, size))
        for i, n in enumerate(ns):
            bars[i, :, ::size//n, ::size//n] = 1.
        yield bars
        # Horizontal bars
        bars = torch.zeros((len(ns), 1, size, size))
        for i, n in enumerate(ns):
            bars[i, :, ::size//n, :] = 1.
        yield bars
        # Vertical bars
        bars = torch.zeros((len(ns), 1, size, size))
        for i, n in enumerate(ns):
            bars[i, :, :, ::size//n] = 1.
        yield bars
        # Lattice bars
        bars = torch.zeros((len(ns), 1, size, size))
        for i, n in enumerate(ns):
            bars[i, :, ::size//n, :] = 1.
            bars[i, :, :, ::size//n] = 1.
        yield bars
    group_info = [
        (name, {'ns': np.array(ns)})
        for name in ['grd', 'hor', 'ver', 'lat']]
    return group_info, group_iter
