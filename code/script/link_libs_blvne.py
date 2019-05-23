import sys, os
sys.path.append('/Users/gru/proj/attfield/code')

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
    code  = _Directory('/Users/gru/proj/attfield/code')
    data  = _Directory('/Users/gru/proj/attfield/data')
    plots = _Directory('/Users/gru/proj/attfield/plots')
__builtins__['Paths'] = Paths