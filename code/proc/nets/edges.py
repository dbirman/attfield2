from torch import nn
import torch

class Identity(nn.Module):
    def forward(self, x): 
        return x

def get_model():
    conv = nn.Conv2d(1, 3, kernel_size = 5, padding = 2,
        stride = 1, bias = None)
    conv.weight.data.zero_()
    # Horizontal edges
    conv.weight.data[0, 0, :2, :] = -1
    conv.weight.data[0, 0, 3:, :] = 1
    # Vertical edges
    conv.weight.data[1, 0, :, :2] = -1
    conv.weight.data[1, 0, :, 3:] = 1
    # Curvature
    conv.weight.data[2, 0, :, :] = -1
    conv.weight.data[2, 0, 2, :] = 1
    conv.weight.data[2, 0, :, 2] = 1

    return nn.Sequential(Identity(), conv)