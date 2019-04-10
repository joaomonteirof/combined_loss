import nevergrad.optimization as optimization
from nevergrad import instrumentation as instru
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
parser.add_argument('--batch-size', type=int, default=24, metavar='N', help='input batch size for training (default: 24)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 16)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--data-path', type=str, default='./data/cifar10_train_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data/cifar10_test_data.hdf', metavar='Path', help='Path to data')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--budget', type=int, default=100, metavar='N', help='Maximum training runs')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

def train(lr, l2, momentum, margin, lambda_, patience, swap, model, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path):

	trainset = Loader(data_path)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=n_workers, worker_init_fn=set_np_randomseed)

	validset = Loader(valid_data_path)
	valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers)

	if model == 'vgg':
		model = vgg.VGG('VGG16')
	elif model == 'resnet':
		model = resnet.ResNet18()
	elif model == 'densenet':
		model = densenet.densenet_cifar()

	if cuda:
		model = model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2, momentum=momentum)

	trainer = TrainLoop(model, optimizer, train_loader, valid_loader, margin=margin, lambda_=lambda_, patience=int(patience), verbose=-1, save_cp=False, checkpoint_path=None, swap=swap, cuda=cuda)

	return trainer.train(n_epochs=epochs)

lr = instru.var.Array(1).asfloat().bounded(1, 4).exponentiated(base=10, coeff=-1)
l2 = instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
momentum = instru.var.Array(1).asfloat().bounded(0.10, 0.95)
margin = instru.var.Array(1).asfloat().bounded(0.10, 1.00)
lambda_ = instru.var.Array(1).asfloat().bounded(1, 5).exponentiated(base=10, coeff=-1)
patience = instru.var.Array(1).asfloat().bounded(1, 100)
swap = instru.var.OrderedDiscrete([True, False])
model = args.model
epochs = args.epochs
batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
n_workers = args.n_workers
cuda = args.cuda
data_path = args.data_path
valid_data_path = args.valid_data_path

instrum = instru.Instrumentation(lr, l2, momentum, margin, lambda_, patience, swap, model, epochs, batch_size, valid_batch_size, n_workers, cuda, data_path, valid_data_path)

hp_optimizer = optimization.optimizerlib.PSO(instrumentation=instrum, budget=args.budget)

print(hp_optimizer.optimize(train))