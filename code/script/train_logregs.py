import importlib.util, os
spec = importlib.util.spec_from_file_location("link_libs", os.environ['LIB_SCRIPT'])
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from proc import detection_task as det
from proc import cornet
from argparse import ArgumentParser

parser = ArgumentParser(
    description = "Train logistic regressions on isolated object detection task.")
parser.add_argument('output_path',
    help = 'Path to an npz file where output should be stored.')
parser.add_argument("iso_images_path",
    help = 'Path to the HDF5 archive containing the input images.')
parser.add_argument("train_n", type = int,
    help = 'Number of images to train regresions on.')
parser.add_argument("decoders", nargs = "+",
    help = 'Layer indices that perform decoding and should be '+
           'skipped when generating encodings. Given as '+
           'list of tuples, e.g. "(0, 4, 2)" for cornet-Z')
parser.add_argument('--model', type = str, default = None,
    help = 'Python file with a function `get_model` that returns a PyTorch'+
           'model for the script to run backprop on. If not provided, the '+
           'script will use CorNet-Z.')
parser.add_argument("--cats", nargs = '*', default = [],
    help = 'If given, a whitelist for categories to train regressors for.')
parser.add_argument('--verbose', action = "store_true",
    help = 'Run with extra progress output.')
args = parser.parse_args()
args.decoders = [eval('tuple('+l+')') for l in args.decoders]
if len(args.cats) == 0: args.cats = None


if args.model is None:
    model, _ = cornet.load_cornet("Z")
else:
    spec = importlib.util.spec_from_file_location(
        "model", args.model)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_model()
iso_task = det.DistilledDetectionTask(
    args.iso_images_path, whitelist = args.cats)
#iso_task = det.FakeDetectionTask(args.cats, 3, 224)

_, _, regs, _, _ = det.fit_logregs(
    model, args.decoders, iso_task,
    train_size = args.train_n,
    shuffle = False,
    verbose = args.verbose)
det.save_logregs(args.output_path, regs)




