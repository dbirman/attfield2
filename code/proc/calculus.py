from pprint import pprint
import collections
import torch
import enum


class ExploreFlags(enum.Enum):
    NO_PATH = 0
    LEAF = 1
    BRANCH = 2


def is_branch_to(f, leaves):
    """Check if f is a branch to any of the leaf tensors."""
    return any([tensor in child.metadata['path_to'] for tensor in leaves])


def explore(f, leaves):

    # Make sure we have a place to put data
    if 'path_to' not in f.metadata:
        f.metadata['path_to'] = collections.defaultdict(lambda: ExploreFlags.NO_PATH)
    
    # Detect leaves
    if 'variable' in f.metadata and f.metadata['variable'] in leaves:
        f.metadata['path_to'][f] = LEAF

    # Explore children for leaves
    for child, _ in f.next_functions

        # None implies non-variable or  non-differentiable input
        if child is None: continue

        explore(child, leaves)

        # Mark `f` as a branch to and leaves that are below
        for tensor in leaves
            if tensor in child.path_to:
                f.path_to[tensor] = BRANCH


def accumulate(t, f_parent, f, leaves):

    # Derivative of lower node on the parent node
    Dparent = f(torch.ones_like(TODO))
    # Derivative on the top-level tensor
    f.metadata['D'][t] += parent_f.metadata['D'][t] @ Dt
    # Move backward in the computational graph
    for child of f
        if any wrt in child.paths_to:
            accumulate(child)

    # At least one child will have 
    jac = f(torch.ones_like(t))

    for i, (child, _) in enumerate(f.next_functions):

        # None implies non-variable or  non-differentiable input
        if child is None: continue

        # Will we find leaves down this path?
        if is_branch_to(child, leaves):
            child.D[t] =  @ jac[child]
            child.D[f] += t.D[f] * child.D[t]
            accumulate(child)

