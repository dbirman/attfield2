import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det
import sys

imagenet_h5 = sys.argv[1]
meta_csv = sys.argv[2]
output_h5 = sys.argv[3]
blacklist = sys.argv[4:]

if __name__ == '__main__':
    det.cache_four_task(
        imagenet_h5,
        output_h5,
        loc = 0,
        blacklist = blacklist,
        metadata_csv = meta_csv)