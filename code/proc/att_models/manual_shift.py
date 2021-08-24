from proc import attention_models as att

def attn_model(layer, beta, r = 0.25, **kws):
    return  {
        tuple(layer): att.GaussianConvShiftAttention((0.25, 0.25), (r, r), beta)
    }