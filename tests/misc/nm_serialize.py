import pickle as pkl
import torch

t = torch.tensor([1, 2, 3, 4], requires_grad = True)
t2 = t**2
s2 = pkl.dumps(t2)
t2_L = pkl.loads(s2)
print(t2_l.grad(torch.ones_like(t2_L)))
