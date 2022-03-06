from torch.utils.data import Dataset
import numpy as np
import torch, os
import vg

class Dataset(Dataset):
    def __init__(self, path, N, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.interpolation = N
        if train:
            self.full_path = [path + "train/" + str(i) + "/" + j for i in os.listdir("path + "train/") for j in os.listdir(path + "train/" + str(i))]
        else:
            self.full_path = [path + "test/" + str(i) + "/" + j for i in os.listdir("path + "test/") for j in os.listdir(path + "test/" + str(i))]
        

    def __len__(self):
        return len(self.full_path)

    def __getitem__(self, idx):
        path = self.full_path[idx]
        data = np.loadtxt(path)
        row, col = np.shape(data)
        data = torch.from_numpy(data)
        data = data.T.reshape((1, col, row))
        data = nn.functional.interpolate(data, size=N).squeeze().T
        data = vg.normalize(data)
        data = data.reshape((data.size(0), col // 3, 3, 1))
        label = torch.tensor([int(self.data_idx[idx][0] - 1)])
        sample = {"data": data, "label": label}
        return sample
