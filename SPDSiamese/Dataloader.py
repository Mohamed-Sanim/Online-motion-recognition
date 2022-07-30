from torch.utils.data import Dataset
import numpy as np
import torch, os
import vg
from torch import nn

class Dataset_learning(Dataset):
    def __init__(self, path, N, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.interpolation = N
        if train:
            self.full_path = [path + "/train/" + str(i) + "/" + j for i in os.listdir(path + "/train/") for j in os.listdir(path + "/train/" + str(i))]
        else:
            self.full_path = [path + "/test/" + str(i) + "/" + j for i in os.listdir(path + "/test/") for j in os.listdir(path + "/test/" + str(i))]
        

    def __len__(self):
        return len(self.full_path)

    def __getitem__(self, idx):
        path = self.full_path[idx]
        data = torch.from_numpy(np.loadtxt(path))
        data = data[1:] - data[:-1]
        row, col = data.size()
        data = data.T.reshape((1, col, row))
        data = nn.functional.interpolate(data, size=self.interpolation, mode='linear').squeeze(0).T
        data = nn.functional.normalize(data)
        data = data.reshape((data.size(0), col // 3, 3, 1))
        label = torch.tensor(int(path.split("/")[-2]))
        sample = {"data": data, "label": label}
        return sample

class Dataset_classification(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        if train:
            x = torch.load(path + "/train.pth")
        else:
            x = torch.load(path + "/test.pth")
        self.data = x['data']
        self.target = x['label']
        
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.target[idx]
        sample = {"data": data, "label": label}
        return sample
