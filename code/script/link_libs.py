import importlib.util, os
if os.environ.get('BLVNE', None) is not None:
    libscript="/Users/gru/proj/attfield/code/script/link_libs_blvne.py"
else:
    libscript="/Users/kaifox/GoogleDrive/attfield/code/script/link_libs_kfmbp.py"
else:
	libscript=""
spec = importlib.util.spec_from_file_location("link_libs", libscript)
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)