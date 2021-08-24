from proc import attention_models as att

def attn_model(layer, beta, r = 0.25, neg_mode = True, **kws):
    '''
    - `neg_mode` --- True for warning, `'raise'` for exception, `'fix'` to offset
        feild locations with a negative to be 0 or positive.
    '''
    # One layer
    if isinstance(layer[0], int):
        return {
            tuple(layer): att.GaussianSensitivityGradAttention((0.25, 0.25), (r, r), beta, neg_mode)
        }
    # Multiple layers
    else:
        return {
            tuple(L): att.GaussianSensitivityGradAttention((0.25, 0.25), (r, r), beta, neg_mode)
            for L in layer
        }