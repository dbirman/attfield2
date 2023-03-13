from proc import attention_models as att

def attn_model(layers, beta, loc = (0.25, 0.25), r = (0.125, 0.125)):
    return  {
        tuple(l): att.GaussianLocatedGain(loc, r, beta)
        for l in layers
    }