import numpy as np
from PIL import Image
import os, sys
import glob
import pandas as pd
from math import ceil
from time import time

import torch
from torch.autograd import Variable
from collections import defaultdict
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data.dataloader import default_collate
import torch.optim as optim
from torch import nn
from torch.distributions import MultivariateNormal


class FewShotSampler:
    def __init__(self, num_cls=200, img_per_class=500, max_batch_cls=None,
                 max_img_per_class=None,
                 separate=False, exact=False,
                 iter=None,
                 timesteps=15, batch_size=64):
        """

        Args:
            num_cls: overall number of classes
            img_per_class (int or iterable with ints): for each class, number of objects in it
            timesteps: number of shots in few-shot learning task
            batch_size: batch size
            exact: whether to threat max_batch_cls or max_img_per_class as
                upper bound on random number (of classes or images resp.)
                or as the exact numbers
            separate: whether to split data into objects for reconstruction and
                few-shots conditions
            max_batch_cls: maximum number of classes in few-shot condition set
            max_img_per_class: maximum number of images per class in few-shot condition set
        """
        if max_batch_cls is None and max_img_per_class is None:
            raise ValueError('specify either max_batch_cls or '
                             'max_img_per_class')
        if max_batch_cls is not None and max_img_per_class is not None:
            raise ValueError('max_batch_cls is muturally exclusive '
                             'with max_img_per_class')
        if exact:
            if max_batch_cls is not None:
                if timesteps % max_batch_cls != 0:
                    raise ValueError('maximum number of classes should '
                                     'be divisor of the number of shots')
            if max_img_per_class is not None:
                if timesteps % max_img_per_class != 0:
                    raise ValueError('maximum number of images per class '
                                     'should be divisor of the number of short')
        self.num_cls = num_cls
        self.separate = separate
        self.iter = iter
        self.exact = exact
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.img_per_class = img_per_class
        self.max_batch_cls = max_batch_cls
        self.max_img_per_class = max_img_per_class
        self.acc = {}
        acc = 0
        if isinstance(self.img_per_class, int):
            imgs_per_class = self.img_per_class
            self.img_per_class = {}
            for i in range(self.num_cls):
                self.img_per_class[i] = imgs_per_class
        self.total = sum(self.img_per_class.values())
        for i, c in self.img_per_class.items():
            self.acc[i] = acc
            acc += c
        self.acc[len(self.img_per_class)] = self.total
        self.classes = np.zeros(sum(self.img_per_class.values()), dtype=np.int32)
        self.objects = {}
        for i in range(len(self.acc) - 1):
            self.classes[self.acc[i]: self.acc[i + 1] + 1] = i
            self.objects[i] = np.arange(self.acc[i], self.acc[i + 1])
        self.stats = defaultdict(int)
        self.stats1 = defaultdict(int)
        self.time = 0

    def __len__(self):
        if self.iter is not None:
            return self.iter
        return int(ceil(self.total / self.batch_size))

    def sample_class(self, clz, max_samples, current=None, exact=False):
        if (not self.exact or self.max_batch_cls is not None) and not exact:
            num_to_sample = np.random.randint(1, max_samples + 1)
        else:
            num_to_sample = max_samples
        img_per_class = self.img_per_class[clz]
        objects = self.objects[clz]
        if current is not None:
            objects = objects[objects != current]
        return np.random.choice(objects, num_to_sample, replace=False).tolist()

    def __iter__(self):
        c = 0
        c1 = 0
        total = sum(self.img_per_class.values())
        perm = np.random.permutation(total)
        while True:
            t = time()
            if c >= len(self):
                break
            conds = []
            if (c1 * self.batch_size) > len(perm):
                c1 = 0
                perm = np.random.permutation(total)
                x_batch = perm[c1 * self.batch_size: (c1 + 1) * self.batch_size]
            else:
                x_batch = perm[c1 * self.batch_size: (c1 + 1) * self.batch_size]
                c1 += 1
            classes = self.classes[x_batch]
            if self.timesteps == 0:
                yield x_batch
                c += 1
                continue
            for j in range(len(x_batch)):
                conds.append([])
                obj = x_batch[j]
                clz = classes[j]
                cls_perm = np.random.permutation(self.num_cls)
                cls_perm = cls_perm[cls_perm != clz]
                if self.max_img_per_class is not None:
                    max_samples = min(self.max_img_per_class, self.timesteps)
                    conds[-1] += self.sample_class(clz, max_samples=max_samples, current=obj)
                    k = 0
                    while len(conds[-1]) != self.timesteps:
                        max_samples = min(self.max_img_per_class, self.timesteps - len(conds[-1]))
                        conds[-1] += self.sample_class(cls_perm[k], max_samples=max_samples)
                        k += 1
                elif self.max_batch_cls is not None:
                    if not self.exact:
                        max_ = min(self.max_batch_cls, self.timesteps)
                        num_cls = np.random.randint(1, max_ + 1)
                    else:
                        num_cls = self.max_batch_cls
                    self.stats1[num_cls] += 1
                    if num_cls == 0:
                        conds[-1] += self.sample_class(clz, self.timesteps,
                                                       current=obj, exact=True)
                    else:
                        max_samples = min(self.timesteps - num_cls, self.img_per_class[clz]) + 1
                        exact = max_samples == self.timesteps
                        conds[-1] += self.sample_class(clz, max_samples, current=obj, exact=exact)
                        k = 0
                        while len(conds[-1]) != self.timesteps:
                            k += 1
                            if k == num_cls:
                                max_samples = self.timesteps - num_cls + k + 1 - len(conds[-1])
                                exact = False
                            else:
                                max_samples = self.timesteps - len(conds[-1])
                                exact = True
                            sampled = self.sample_class(cls_perm[k - 1], max_samples, exact=exact)
                            conds[-1] += sampled

                        self.stats[k + 1] += 1
                conds[-1] = np.array(conds[-1])
            conds = np.row_stack(conds)
            self.time += time() - t
            yield np.column_stack([conds, x_batch]).flatten()
            c += 1


class FewShotCollate:
    def __init__(self, timesteps, device=0):
        self.timesteps = timesteps
        self.device = device

    def __call__(self, batch):
        data, labs = default_collate(batch)
        #data = data.to(self.device, non_blocking=True)
        #labs = labs.to(self.device, non_blocking=True)
        data = data.view(data.shape[0], -1)
        x_dim = data.shape[1]
        timesteps = self.timesteps
        batch_size = data.shape[0] // (timesteps + 1)
        data = data.view(batch_size, timesteps + 1, x_dim)
        labs = labs.view(batch_size, timesteps + 1)
        conds, x = data[:, :-1].contiguous(), data[:, -1].contiguous()
        cond_labs, labs = labs[:, :-1].contiguous(), labs[:, -1].contiguous()
        return (conds, x), (cond_labs, labs)


def few_shot_mnist(root, train=True, batch_size=128, timesteps=15, n_jobs=0, **kwargs):
    train_set = dset.MNIST(
        root=root, train=train, download=True,
        transform=transforms.ToTensor(),
    )
    sort = train_set.targets.argsort()
    train_set.targets = train_set.targets[sort].contiguous()
    train_set.data = train_set.data[sort].contiguous()
    # counts of images per class
    cnts = (pd.Series(train_set.targets.numpy())
            .to_frame('cls').reset_index()
            .groupby('cls')['index'].count()
            .to_dict())
    collate = FewShotCollate(timesteps=timesteps)
    batch_sampler = FewShotSampler(
        num_cls=10, img_per_class=cnts, separate=True,
        batch_size=batch_size, timesteps=timesteps, **kwargs
    )
    return torch.utils.data.DataLoader(
        dataset=train_set, batch_sampler=batch_sampler,
        collate_fn=collate, num_workers=n_jobs
    )


def few_shot_omniglot(root, train=True, batch_size=128, timesteps=15,
                      n_jobs=0, resize=28, **kwargs):
    train_set = dset.Omniglot(root=root, download=True, background=train,
                              transform=transforms.Compose([
                                  transforms.Resize([resize, resize], interpolation=Image.NEAREST),
                                  transforms.ToTensor(),
                                  transforms.Lambda(lambda x: 1 - x)
                              ]))
    collate = FewShotCollate(timesteps=timesteps)
    sampler = FewShotSampler(
        num_cls=len(train_set._character_images),
        batch_size=batch_size, separate=True,
        timesteps=timesteps, **kwargs,
        img_per_class=len(train_set._character_images[0]),
    )
    return torch.utils.data.DataLoader(
        dataset=train_set, batch_sampler=sampler,
        collate_fn=collate, num_workers=n_jobs
    )
