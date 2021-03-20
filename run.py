import torch
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from argparse import ArgumentParser
from tensorboardX import SummaryWriter

from trainer import FewShotTrainer
from parsers import Experiment

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--logdir', type=str, default='')
args = parser.parse_args()
if not (args.train or args.test):
    raise ValueError('Specify either --train or --test')
logdir = args.logdir
base, ldir = os.path.split(logdir)
train_dir = os.path.join(base, 'train', ldir)
test_dir = os.path.join(base, 'test', ldir)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
with torch.autograd.detect_anomaly():
    train_writer = SummaryWriter(logdir=train_dir)
    test_writer = SummaryWriter(logdir=test_dir)
    exp = Experiment(args.config, train=args.train)
    trainer = FewShotTrainer(exp, train_writer=train_writer, test_writer=test_writer)
    trainer.run()
