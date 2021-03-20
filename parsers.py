import os
import yaml
import logging
import torch
from copy import copy
from math import ceil
from itertools import product
from models import VMA
from autoencoders import VMAEncoder, VMADecoder
from dataloaders import few_shot_mnist, few_shot_omniglot

logger = logging.getLogger(__name__)


def get_constructor(dataset):
    if dataset == 'mnist':
        return few_shot_mnist
    elif dataset == 'omniglot':
        return few_shot_omniglot
    elif dataset == 'imgnet':
        return few_shot_imgnet
    else:
        raise ValueError(f'unsupported dataset name: {dataset}')


def get_dataloader(data_config, n_jobs=0, train=True, **kwargs):
    if not isinstance(data_config, tuple):
        data_config = (data_config,)
    if len(data_config) > 1:
        if not isinstance(train, (list, tuple)):
            train = [train] * len(data_config)
    else:
        if not isinstance(train, (list, tuple)):
            train = [train]
    if len(train) > len(data_config):
        raise ValueError('Too many train flags')
    dataloaders = []
    for cfg, train_flag in zip(data_config, train):
        train_dl = get_constructor(cfg['kind'])
        del cfg['kind']
        for k, v in kwargs.items():
            if k in cfg:
                cfg[k] = v
        train_dl = train_dl(n_jobs=n_jobs, train=train_flag, **cfg)
        dataloaders.append(train_dl)
    if len(dataloaders) == 1:
        return dataloaders[0]
    return dataloaders


# todo: remove "data dim"
def get_model(model_config, data_dim=784):
    mc = model_config
    kind = mc['kind']
    autoencoder = mc['autoencoder']
    kinds = {
        'VMA': VMA,
        # 'GMN': GMN,
        # 'NS': NeuralStatistician
    }
    autoencoders = {
        'VMA': (VMAEncoder, VMADecoder),
    }
    if kind not in kinds:
        raise ValueError(f'Unsupported model name: {kind}')
    if autoencoder not in autoencoders:
        raise ValueError(f'Unsupported autoencoder: {autoencoder}')

    model = kinds[kind]
    encoder, decoder = autoencoders[autoencoder]
    del mc['autoencoder']
    del mc['kind']
    return model(encoder=encoder, decoder=decoder, data_dim=data_dim, **mc)


class Experiment:
    def __init__(self, config, train=True):
        if isinstance(config, str):
            self.path = os.path.join(os.path.split(config)[0])
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        cfg = config
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.checkpoint_path = None
        self.importance_num = None
        self.train_loader = None
        self.output_file = None
        self.val_loader = None
        self.optimizer = None
        self.multi_gpu = None
        self.full_loss = None
        self.scheduler = None
        self.log_file = None
        self.loaders = None
        self.epochs = None
        self.params = None
        self.metric = None
        self.model = None
        self.gpu = None

        if self.mode == 'train':
            self.get_train_spec(cfg)
        elif self.mode == 'test':
            self.get_test_spec(cfg)

    def __getattr__(self, name):
        return None

    def get_train_spec(self, cfg):
        self.resize = cfg['train_data']['resize']
        self.model = get_model(cfg['model'], data_dim=self.resize ** 2)
        loaders = get_dataloader((cfg['train_data'], cfg['val_data']),
                                 train=[True, False],
                                 n_jobs=cfg['train_spec']['n_jobs'])
        self.train_loader, self.val_loader = loaders
        cfg = cfg['train_spec']
        self.gpu = [int(g) for g in cfg['gpus']]
        self.model.to(self.gpu[0])
        self.multi_gpu = len(self.gpu) > 1
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu)
        self.epochs = int(ceil(int(float(cfg['steps'])) / len(self.train_loader)))
        optimizer = getattr(torch.optim, cfg['optimizer'])
        self.scheduler = None
        wd = 0.
        if 'weight_decay' in cfg:
            wd = float(cfg['weight_decay'])
        self.optimizer = optimizer(self.model.parameters(), lr=float(cfg['lr']),
                                   weight_decay=wd)
        if 'scheduler' in cfg:
            assert 'kind' in cfg['scheduler']
            sched = getattr(torch.optim.lr_scheduler, cfg['scheduler']['kind'])
            ep_steps = len(self.train_loader)
            step_size = cfg['scheduler']['step_size'] // ep_steps
            gamma = cfg['scheduler']['gamma']
            self.scheduler = sched(self.optimizer, step_size, gamma)
        self.log_file = cfg['log_file']
        self.full_loss = cfg.get('full_loss', False)
        self.checkpoint_path = cfg['checkpoint_path']

    def _parse_list(self, values):
        result = []
        for v in values:
            if isinstance(v, str) and '..' in v:
                start, end = v.split('..')
                result += list(range(int(start), int(end) + 1))
            else:
                result.append(int(v))
        return result

    def get_test_spec(self, cfg):
        self.resize = cfg['test_data']['resize']
        self.model = get_model(cfg['model'], data_dim=self.resize ** 2)
        state = torch.load(cfg['test_spec']['checkpoint_path'])
        self.model.load_state_dict(state)
        test_data_cfg = cfg['test_data']
        cfg = cfg['test_spec']
        self.gpu = [int(g) for g in cfg['gpus']]
        self.model.to(self.gpu[0])
        self.multi_gpu = len(self.gpu) > 1
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu)
        params = cfg['test_params']
        self.loaders = []
        for p, vs in params.items():
            params[p] = self._parse_list(vs)
        self.params = [dict(zip(params.keys(), vs)) for vs in product(*params.values())]
        for vs in self.params:
            self.loaders.append(
                get_dataloader(copy(test_data_cfg), n_jobs=cfg['n_jobs'],
                               train=False, **vs)
            )
        self.metric = cfg['metric']
        self.output_file = cfg['output_file']
        self.importance_num = cfg['importance_num']
