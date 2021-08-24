import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from skimage import io as skio
from scipy import io as scio
import numpy as np
import h5py


imgs = scio.loadmat('data/imagenet_human/stimuli.mat')
skio.imsave('plots/figures/fig1/wheel_positive_dist.png', imgs['data'][1449])
skio.imsave('plots/figures/fig1/wheel_negative_dist.png', imgs['data'][1485])
skio.imsave('plots/figures/fig1/wheel_positive_focl.png', imgs['data'][1460])
skio.imsave('plots/figures/fig1/wheel_negative_focl.png', imgs['data'][1494])

noise = np.random.uniform(low=0, high=255, size=(224, 224, 3)).astype('uint8')
skio.imsave('plots/figures/fig1/noise.png', noise)
