import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import voxel_selection as vx
from proc import network_manager as nm
from proc import backprop_fields
from proc import possible_fields
from proc import lsq_fields
from proc import video_gen
from proc import cornet

from proc import detection_task as det
import plot.behavioral as plot_bhv

from pprint import pprint
import numpy as np
import skvideo.io


if __name__ == '__main__':
	LAYERS = [(0, 0, 1)]

	model, ckpt = cornet.load_cornet("Z")
	inputs = video_gen.noise_video(11, 3, 64, 64, numpy = False)
	voxels = vx.random_voxels_for_model(model, LAYERS, 10, 3, 64, 64)

	inputs = {"noise": inputs}
	regs = {"noise": det.ClassifierMod(np.random.randn(512), 0)}
	vgrads = det.voxel_decision_grads(voxels, model, [(0, 4, 2)], inputs, regs)
