from proc import attention_models as att

def attn_model(layer, beta, loc = (0.25, 0.25), r = (0.25, 0.25)):
    return  {
        tuple(layer): att.GaussianLocatedAdd(loc, r, beta)
    }