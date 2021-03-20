from visdom import Visdom
import numpy as np
import math
from collections import defaultdict


def draw(imgs, color=True, nr=None, nc=None):
    import matplotlib.pyplot as plt
    if nr is None:
        size = imgs.shape[0]
        size = int(math.ceil(np.sqrt(size)))
        nr = nc = size
    
    if color:
        w, h, c = imgs.shape[1:]
        plane = np.zeros((w * nr,  h * nc, 3))
    else:
        w, h = imgs.shape[1:]
        plane = np.zeros((w * nr,  h * nc))
    for i in range(nr):
        for j in range(nc):
            if i * nc + j < len(imgs):
                plane[i * h: (i + 1) * h,j * w: (j + 1) * w] = imgs[i * nc + j]
    plt.imshow(plane)

class Drawer:
    def __init__(self, name='evn'):
        self.vis = Visdom(env=name)
        self.name = name
        self.data = defaultdict(list)
        
    def add_value(self, key, value, update='append', name='', alpha=0.02):
        if hasattr(value, 'item'):
            value = value.item()
        self.data[key].append(value)
        ys = Drawer.smooth(np.array(self.data[key])) if len(self.data[key]) > 200 else np.array(self.data[key])
        self.vis.line(
            X=np.arange(len(ys)),
            Y=ys,
            opts={
                'title': key,
                'xlabel': 'step'},
            win=key, env=self.name)
    
    @staticmethod
    def smooth(x, alpha=0.02):
        n = int(alpha * len(x))
        k = n // 2 if n % 2 != 0 else (n // 2) - 1
        ks = n - int(n % 2 == 0)
        cs = x[:ks].mean()
        cs2 = x[-ks:].mean()
        return np.convolve(np.pad(x, k, mode='constant', constant_values=(cs, cs2)), np.ones(ks) / ks, mode='valid')    
        
    def add_images(self, key, images, nrow=None):
        if nrow is None:
            nrow = np.sqrt(len(images))
            nrow = math.ceil(nrow)
            nrow = int(nrow)
        assert images.ndim == 4
        self.data[key] = images
        self.vis.images(images, opts={'title': key}, nrow=nrow, win=key, env=self.name)