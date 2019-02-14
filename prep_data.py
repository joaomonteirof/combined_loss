import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import argparse
import os
import h5py

def prepare_data(data_path, test_data):

	transform = transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	images = datasets.CIFAR10(root=data_path, train=not test_data, download=True, transform=transform)

	return images

def create_dataset(samples, out_file):

	hdf = h5py.File(out_file, 'a')

	for i in range(len(samples)):
		print(i)
		img, label = samples[i]

		img = np.expand_dims(img, 0)
		label = str(label)

		try:
			hdf[label].resize(hdf[label].shape[0]+img.shape[0], axis=0)
			hdf[label][-img.shape[0]:]=img
		except KeyError:
			hdf.create_dataset(label, data=img, maxshape=(None, img.shape[1], img.shape[2], img.shape[3]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Store cifar into hdf indexing by class')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-file', type=str, default='./cifar10_train_data.hdf', metavar='Path', help='output path')
	parser.add_argument('--test-data', action='store_true', default=False, help='Generates hdf for test data')
	args = parser.parse_args()

	if args.test_data:
		print('Preparing test data')
	else:
		print('Preparing train data')

	if os.path.isfile(args.out_file):
		os.remove(args.out_file)
		print(args.out_file+' Removed')

	full_data = prepare_data(args.data_path, args.test_data)
	create_dataset(full_data, args.out_file)
