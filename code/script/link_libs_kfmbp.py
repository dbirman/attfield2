import sys, os
sys.path.append('/Users/kaifox/projects/art_physio/code')

class _Directory:
    def __init__(self, path):
        self.path = path
    def join(self, *args, **kwargs):
        return os.path.join(self.path, *args, **kwargs)
    def __call__(self, *args, **kwargs):
        return os.path.join(self.path, *args, **kwargs)
    def __add__(self, other):
        return os.path.join(self.path, other)
    def __str__(self):
        return self.path

class Paths:
    code  = _Directory('/Users/kaifox/projects/art_physio/code')
    data  = _Directory('/Users/kaifox/projects/art_physio/data')
    plots = _Directory('/Users/kaifox/projects/art_physio/plots')
__builtins__['Paths'] = Paths