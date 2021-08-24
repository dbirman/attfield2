from proc import attention_models as att

def attn_model(layer, beta, loc = (0.25, 0.25), r = (0.125, 0.125)):
    return  {
        tuple(layer): att.GaussianLocatedGain(loc, r, beta)
    }