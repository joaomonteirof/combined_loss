import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Loader(Dataset):

	def __init__(self, hdf5_name, n_cycles=300):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.n_cycles = n_cycles

		open_file = h5py.File(self.hdf5_name, 'r')
		self.classes_list = list(open_file.keys())
		open_file.close()
		self.open_file = None

		self.class2label = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

	def __getitem__(self, index):

		class_idx = index % len(self.classes_list)

		class_ = self.classes_list[class_idx]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		idxs, samples = np.random.choice(np.arange(self.open_file[class_].shape[0]), replace=False, size=5), []

		for idx in idxs:
			sample = self.open_file[class_][idx]
			samples.append( torch.from_numpy( sample ).unsqueeze(0).float().contiguous() )

		return torch.cat(samples, 0), torch.LongTensor(5*[self.class2label[class_]])

	def __len__(self):
		return len(self.classes_list)*self.n_cycles
