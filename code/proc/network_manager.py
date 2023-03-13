from pprint import pprint
from torch import nn
import torch

NULL_HOOK = lambda x: None

class LayerMod(nn.Module):
    
    """Base LayerMod class implementing an identity modification."""
    
    def pre_layer(self, *args, **kwargs):
        """Transform to the inputs before the computation done by
        the layer or its children.
        ### Returns
        - `args`,`kwargs` --- (Possibly) edited versions of the
            inputs that will then be passed to the layer's call
            method.
        - `cache` -- An object to pass to self.post_layer once
            the layer's call method has completed."""

        return args, kwargs, None
    def post_layer(self, outputs, cache):
        """Transform to be applied after the NetworkManager has
        recorded output from the layer."""
        return outputs



class NetworkManager(object):
    __curr_mgr = None # Temporary until NullManager is defined
    __mgr_hist = []
    
    def __init__(self, mods = {}, with_grad = False, cuda = False):
        self.full_idx = ()
        self.curr_idx = 0
        self.computed = {}
        self.modules = {}
        self.grads = {}
        self.nonleaf_hooks = {}
        self.mods = mods
        self.saved_curr_idx = []
        self.with_grad = with_grad
        self.cuda = cuda


    def pre_forward(self, args, kwargs):
        """Register movement INTO a child branch of the computation tree.
        ### Arguments
        - `args`, `kwargs` --- Arguments given to the wrapped Module.__call__
            method that spawned this pre_forward.
        ### Returns
        - `args`, `kwargs` --- Potentially modified arguments to be given 
            to the raw Module.__call__ method that spawned this pre_forward.
        """
        # Denote the current location in the tree as the child branch
        # into which we're delving
        self.full_idx = self.full_idx + (self.curr_idx,)
        
        # Save where we are within our neighbor nodes
        # Then reset that context for our children and recurse
        self.saved_curr_idx.append(self.curr_idx)
        self.curr_idx = 0
        
        # Apply any mods to the layer inputs
        cache = None
        for mod in self.mods_to_apply(self.full_idx):
            args, kwargs, cache = mod.pre_layer(*args, **kwargs)
        return args, kwargs, cache
    
    def post_forward(self, module, outputs, cache):
        """Register movement OUT OF a child branch of the computation tree
        and record the results of the computation."""
        # Returning to ourselves, restore and increment our
        # local potition within our neighbor nodes
        self.curr_idx = self.saved_curr_idx.pop(-1) + 1

        # Apply any mods to the layer outputs
        for mod in self.mods_to_apply(self.full_idx):
            outputs = mod.post_layer(outputs, cache)
        if isinstance(outputs, torch.Tensor) and self.with_grad:
            self._require_nonleaf_grad(outputs, self.full_idx)

        # Bind the computational graph to involved tensors
        # This is used in the custom backprop
        '''
        if self.with_grad:
            if isinstance(outputs, tuple):
                for t in outputs:
                    if t.grad_fn is not None:
                        t.grad_fn.metadata['variable'] = t
            else:
                outputs.grad_fn.metadata['variable'] = outputs
        '''

        # Log results of copmutation done as this step
        self.computed[self.full_idx] = outputs
        self.modules[self.full_idx] = module
        
        # Denote the fact that we are exiting this branch
        self.full_idx = self.full_idx[:-1]
        
        return outputs
    
    def _require_nonleaf_grad(self, t, key):
        '''Apply torch hook to force the tensor `t` to have its gradients
        calculated during a backprop.'''
        def hook(g):
            g._isnlg = True
            self.grads[key] = g
        self.nonleaf_hooks[key] = t.register_hook(hook)

    def close_hooks(self):
        for hook in self.nonleaf_hooks.values():
            hook.remove()

    
    def mods_to_apply(self, layer_idx):
        # Put mods on GPU if requested
        if self.cuda:
            D = lambda t: t.cuda()
        else:
            D = lambda t: t.cpu()

        if layer_idx in self.mods:
            if isinstance(self.mods[layer_idx], LayerMod):
                return (D(self.mods[layer_idx]),)
            else:
                # self.mods[layer_idx] will be of type tuple otherwise
                return tuple(D(m) for m in self.mods[layer_idx])
        else:
            return ()
    
    
    def __enter__(self):
        NetworkManager.__mgr_hist.append(NetworkManager.__curr_mgr)
        NetworkManager.__curr_mgr = self
        return self
    
    def __exit__(self, type, value, traceback):
        NetworkManager.__curr_mgr = NetworkManager.__mgr_hist.pop(-1)
        
    
    @staticmethod
    def current():
        return NetworkManager.__curr_mgr

    @staticmethod
    def assemble(model, inputs, mods = {}, with_grad = False, cuda = False):

        mgr = NetworkManager(mods = mods, with_grad = with_grad, cuda = cuda)
        if with_grad:
            inputs.requires_grad_()
        with mgr:
            if cuda:
                model.cuda()
                inputs = inputs.cuda()
            computed = model(inputs)
        return mgr


    def summarize(self, level = None):
        for k in self.modules:
            if level is None or len(k) <= level:
                print(k, ":", str(self.modules[k]) + " => " +
                              summstr(self.computed[k])
                )


    def save(filename):
        # Serialize piecewise for memory's sake
        pass
        

def mod_merge(*mods):
    ret = {}
    for m in mods:
        for k in m:
            if k in ret:
                if isinstance(ret[k], LayerMod):
                    ret[k] = (ret[k], m[k])
                else:
                    ret[k] = ret[k] + m[k]
            else:
                ret[k] = m[k]
    return ret


def summstr(j):
    return summstr.printmethods[type(j)](j)
summstr.printmethods = {
    torch.Tensor: lambda j: str(j.shape),
    tuple: lambda j: "(" + ','.join(summstr(i) for i in j) + ")"
}


class NullManager(NetworkManager):
    def pre_forward(self, a, kw): return a, kw, None
    def post_forward(self, m, o, c): return o
NetworkManager.__curr_mgr = NullManager()





def patch(klass):
    """Monkeypatching function.
    If subclasses of nn.Module are implemented that override the
    __call__ method then patch(nn.Module) should be called after
    the subclasses are defined."""

    # If the subclass doesn't reimplement __call__ then the function
    # will be identical to that of a superclass and so we don't want to
    # re-wrap. Additionally if this is called twice during program 
    # execution we don't ever want to doubly wrap a __call__ method
    # but still will want to explore the subclass tree
    if not hasattr(klass.__call__, 'old'):

        def call_wrapper(self, *args, **kwargs):
            mgr = NetworkManager.current()
            args, kwargs, cache = mgr.pre_forward(args, {**kwargs, '__layer': self})
            if '__layer' in kwargs: kwargs.pop('__layer')
            ret = call_wrapper.old(self, *args, **kwargs)
            ret = mgr.post_forward(self, ret, cache)
            return ret

        call_wrapper.old = klass.__call__
        klass.__call__ = call_wrapper
    
    for subklass in klass.__subclasses__():
        patch(subklass)

patch(nn.Module)

