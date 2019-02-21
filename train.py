from __future__ import print_function
import argparse
import torch
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
from data_load import Loader
import numpy as np

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

# Training settings
parser = argparse.ArgumentParser(description='Cifar10 Classification')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--l2', type=float, default=5e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='lambda', help='Momentum (default: 0.9)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data/cifar10_train_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/cifar10_test_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before logging training status. Default is 1')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

trainset = Loader(args.data_path)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed)

validset = Loader(args.valid_data_path)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.n_workers)

if args.model == 'vgg':
	model = vgg.VGG('VGG16')
elif args.model == 'resnet':
	model = resnet.ResNet18()
elif args.model == 'densenet':
	model = densenet.densenet_cifar()

if args.cuda:
	model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2, momentum=args.momentum)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))
print('Selected model: {}'.format(args.model))
print('Batch size: {}'.format(args.batch_size))
print('LR: {}'.format(args.lr))
print('Momentum: {}'.format(args.momentum))
print('l2: {}'.format(args.l2))

trainer.train(n_epochs=args.epochs, save_every=args.save_every)
