import importlib.util
import os.path
import torch

def load_cornet(model_type, root = "data/models"):

    spec = importlib.util.spec_from_file_location(
        "cornet_" + model_type,
        os.path.join(root, "cornet_" + model_type.lower() + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, f'CORnet_{model_type.upper()}')()

    ckpt_data = torch.load(
        os.path.join(root, "cornet_" + model_type.upper() + '.pth.tar'),
        map_location=lambda storage, loc: storage)
    model.load_state_dict({
        '.'.join(k.split(".")[1:]): v
        for k,v in ckpt_data['state_dict'].items()})

    return model, ckpt_data