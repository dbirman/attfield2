import sys, os
sys.path.append('/content/gdrive/My Drive/attfield/code')

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
    code  = _Directory('/content/gdrive/My Drive/attfield/code')
    data  = _Directory('/content/gdrive/My Drive/attfield/data')
    plots = _Directory('/content/gdrive/My Drive/attfield/plots')
__builtins__['Paths'] = Paths