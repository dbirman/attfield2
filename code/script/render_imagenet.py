import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det
import sys

index_file = sys.argv[1]
imagenet_h5 = sys.argv[2]
blacklist = sys.argv[3:]

if __name__ == '__main__':
	det.generate_dataset(
		index_file,
		imagenet_h5,
		blacklist = blacklist)

# if __name__ == '__main__':
# 	det.generate_dataset(
# 		Paths.data('imagenet/index.csv'),
# 		Paths.data('imagenet.h5'))