from proc import detection_task as det

def attn_model(layer, beta):
	raise NotImplementedError("Deprecated")
    return  {
        tuple(layer): det.QuadAttention(beta, 0, profile = 'gauss')
    }