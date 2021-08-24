import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

import proc.detection_task as det
from proc import video_gen

import matplotlib.pyplot as plt
import skimage.io
import sys

PATH = sys.argv[1]
CAT = sys.argv[2]
IMG = int(sys.argv[3])
OUT = sys.argv[4]
task = det.DistilledDetectionTask(PATH)
val_imgs, val_ys, _ = task.train_set(CAT, IMG)
print(video_gen.to_numpy(val_imgs)[-1].shape)
skimage.io.imsave(OUT, video_gen.to_numpy(val_imgs)[-1])