import os
from math import ceil
from tqdm import tqdm
from utils_ import to
import numpy as np
import torch
import pandas as pd
from pprint import pprint
from collections import defaultdict
from scipy.special import logsumexp


class FewShotTrainer:
    def __init__(self, exp, train_writer=None, test_writer=None, generate_every=10, test_every=10):
        self.model = exp.model
        self.optimizer = exp.optimizer
        self.scheduler = exp.scheduler
        self.exp = exp
        self.full = exp.full_loss
        self.step = 0
        self.epoch = 0
        self.test_every = test_every
        self.train_writer = train_writer
        self.val_writer = test_writer
        self.generate_every = generate_every
        self.device = exp.gpu[0]
        print(self.model)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def module(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def l2_penalty(self):
        loss = 0.
        for p in self.model.parameters():
            loss += torch.norm(p)
        return loss

    def save_model(self, path):
        self.module.cpu()
        path_dir, path_file = os.path.split(path)
        if len(path_dir) != 0 and not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.module.state_dict(), path)
        self.module.to(self.device)

    def run(self, verbose=True):
        if self.exp.mode == 'train':
            print(f'total epochs: {self.exp.epochs}')
            for i in range(self.exp.epochs):
                self.train_epoch(self.exp.train_loader, verbose=verbose,
                                 checkpoint_path=self.exp.checkpoint_path,
                                 test_loader=self.exp.test_loader)
                if i % self.test_every == 0:
                    train_loss = self.test_epoch(self.exp.train_loader, verbose=verbose, iw=None)
                    test_loss = self.test_epoch(self.exp.val_loader, verbose=verbose,)
                    self.val_writer.add_scalar('ELBO', test_loss, self.step)
                    print(f'train ELBO: {train_loss:.4f}, val ELBO: {test_loss:.4f}')
        elif self.exp.mode == 'test':
            params = []
            for test_loader, p in zip(self.exp.loaders, self.exp.params):
                metrics = self.test_epoch(test_loader, iw=self.exp.importance_num,
                                         verbose=verbose)
                if not isinstance(metrics, dict):
                    metrics = {'ELBO': metrics}
                params.append(p)
                for n, metric in metrics.items():
                    params[-1][n] = metric
                pprint(p)
            pd.DataFrame.from_records(params).to_csv(self.exp.output_file)

    def train_epoch(self, dataloader, verbose=True,
                    checkpoint_path=None, test_loader=None):
        self.model.train(True)
        tq = tqdm(total=len(dataloader), disable=not verbose)
        for j, data in enumerate(dataloader, start=1):
            self.optimizer.zero_grad()
            data = to(data, self.device)
            curr_loss, hist = self.model(data)
            loss = curr_loss.mean()
            for k in hist:
                hist[k] = hist[k].mean()
            if np.isnan(loss.item()):
                raise ValueError()
            loss.backward()
            self.optimizer.step()
            if j % self.generate_every == 0:
                gen_n = 25
                gen = self.module.generate_batch(data)[:gen_n]
                size = self.exp.resize
                gen = gen.view(-1, 1, size, size)
                gen = gen[:gen_n]
                gen = (gen).detach().cpu().numpy()
                self.train_writer.add_images('generated', gen, self.step)
            for k in hist:
                self.train_writer.add_scalar(k, hist[k], self.step)
            tq.update(1)
            self.step += 1
        if test_loader is not None:
            self.model.train(False)
            loss = 0.
            agg = defaultdict(float)
            for k, data in enumerate(test_loader, start=1):
                data = to(data, self.device)
                curr_loss, hist = self.model(data, test=True)
                loss += curr_loss.mean().mean().item()
                for k in hist:
                    agg[k] += hist[k].mean().item()
            for k in agg:
                agg[k] = agg[k] / len(test_loader)
                self.val_writer.add_scalar(k, agg[k], self.step)
            self.model.train(True)
        if checkpoint_path is not None:
            self.save_model(checkpoint_path)
        if self.scheduler is not None:
            self.scheduler.step()
        tq.close()

    def test_epoch(self, test_dataloader, iw=50, iw_cond=6, verbose=True):
        self.model.train(False)
        with torch.no_grad():
            metrics = defaultdict(float)
            tq = tqdm(total=len(test_dataloader), disable=not verbose)
            elbo = 0
            for j, data in enumerate(test_dataloader, start=1):
                history = {}
                data = to(data, self.device)
                (_, x_u), _ = data
                l, h = self.model(data, iw=iw, test=True, mean=False)
                elbo += l.mean().item()
                tq.update(1)
            elbo = elbo / len(test_dataloader)
            tq.close()
            return elbo
