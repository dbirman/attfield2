import importlib.util
spec = importlib.util.spec_from_file_location("link_libs",
    "/Users/kaifox/projects/art_physio/code/script/link_libs_kfmbp.py")
link_libs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link_libs)

from torch import nn
import torch
import proc.network_manager as netmgr
from pprint import pprint

class MultMod(netmgr.LayerMod):
    def __init__(self, vox_idx, value):
        self.vox_idx = vox_idx,
        self.value = value
    def pre_layer(self, inp):
        print("Running MultMod.pre")
        mod = torch.ones_like(inp)
        mod[self.vox_idx] = self.value
        # Returns must be in args, kwargs format
        return (inp * mod,), {}


class Identity(nn.Module):
    def forward(self, x):
        print("running Indentity.forward")
        return x

class TestBlock(nn.Module):

    def __init__(self):
        super(TestBlock, self).__init__()
        self.l1 = Identity()
        self.l2 = Identity()

    def forward(self, inp):
        x1 = self.l1(inp)
        x2 = self.l2(x1)
        return x2



if __name__ == '__main__':
    net = TestBlock()
    inp = torch.ones([5])
    mgr = netmgr.NetworkManager(mods = {
        (0, 1): MultMod(1, 5)
    })
    with mgr:
        computed = net(inp)
        print(mgr.full_idx, mgr.curr_idx)
    print("\n[[====  Computed  ======================]]")
    print(computed)
    
    print("\n[[====  Layer-Wise Computed  ===========]]")
    pprint(mgr.computed)
    
    print("\n[[====  Module Structure  ==============]]")
    pprint(mgr.modules)