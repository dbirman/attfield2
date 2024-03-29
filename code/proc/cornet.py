import importlib.util
import os.path
import torch


def load_cornet(model_type,
    model = None,
    code_root = Paths.code.join('cornet/cornet'),
    weight_root = Paths.data.join("models")):
    

    '''spec = importlib.util.spec_from_file_location(
        "cornet_" + model_type,
        os.path.join(code_root, "cornet_" + model_type.lower() + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, f'CORnet_{model_type.upper()}')()'''
    # If model not given then load one according the given cornet type name
    if model is None:
        if model_type == 'Z':
            from cornet.cornet import cornet_z
            model = cornet_z.CORnet_Z()
        elif model_type == 'S':
            from cornet.cornet import cornet_s
            model = cornet_s.CORnet_S()

    ckpt_data = torch.load(
        os.path.join(weight_root, "cornet_" + model_type.upper() + '.pth.tar'),
        map_location=lambda storage, loc: storage)
    model.load_state_dict({
        '.'.join(k.split(".")[1:]): v
        for k,v in ckpt_data['state_dict'].items()})

    return model, ckpt_data