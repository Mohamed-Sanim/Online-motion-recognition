import torch
from torch import nn
import numpy as np
import random
import os

def makedir_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
def set_dataset(dataset, execution, path, primary_path, ws, m, type_detector):
    if execution == "Classifier":
        for i in os.listdir(primary_path):
            makedir_path(path + "/" + i)
            for j in os.listdir(primary_path + i):
                c = np.loadtxt(primary_path + i + "/" + j + "/groundtruth.txt")
                data = np.loadtxt(primary_path + i + "/" + j + "/skeletal_sequence.txt")
                for k in range(len(c)):
                    makedir_path(path + "/" + i + "/" + str(int(c[k, 0])))
                    np.savetxt(path + "/" + i + "/" + str(int(c[k, 0])) + "/" + j + "_" + str(int(c[k, 1])) \
                        + "_"+ str(int(c[k, 2])) + ".txt", data[int(c[k, 1]): int(c[k, 2]) + 1], fmt='%.6f')
    elif type_detector[dataset] == "Binary":
        limit = {"train": 5000, "test": 3000}
        for i in os.listdir(primary_path):
            makedir_path(path + "/" + i)
            makedir_path(path + "/" + i + "/0")
            makedir_path(path + "/" + i + "/1")
            for j in os.listdir(primary_path + i):
                c = torch.from_numpy(np.loadtxt(primary_path + i + "/" + j + "/groundtruth.txt")[:,1:]).type(torch.LongTensor)
                t = int(c[0,0])
                data = np.loadtxt(primary_path + i + "/" + j + "/skeletal_sequence.txt")[t:]
                c -= t
                c = torch.cat((c, c[:,1:] + 1, torch.cat((c[1:,0:1] - 1, torch.tensor([[len(data)]])))), 1).reshape(len(c) * 2, 2).type(torch.LongTensor)
                d = torch.LongTensor([[k * m, k * m + ws] for k in range((len(data) - ws) // m)])
                cn = [set(range(int(c[k,0]), int(c[k,1]))) for k in range(len(c))]
                dn = [set(range(int(d[k,0]), int(d[k,1]))) for k in range(len(d))]
                vndn = [[np.round(len(cn[k].intersection(dn[df]))/ws * 100) 
                        for df in range(len(dn))] for k in range(len(cn))]
                cd = (torch.tensor(vndn).max(0)[1] + 1) % 2
                for k in range(len(d)):
                    np.savetxt(path + "/" + i + "/" + str(int(cd[k])) + "/" + j + "_" + str(int(d[k,0] + t))
                        + "_" + str(int(d[k,1] + t)) + ".txt" , data[d[k,0]:d[k,1]], fmt="%.6f")
            for lab in ["0", "1"]:
                c0 = os.listdir(path + "/" + i + "/" + lab)
                if (i == "train" and len(c0) > limit[i]) or (i == "test" and len(c0) > limit[i]):
                    for ii in random.sample(c0, len(c0) - limit[i]):
                        os.remove(path + "/" + i + "/" + lab + "/" + ii)
    else:
        for i in os.listdir(primary_path):
            makedir_path(path + "/" + i)
            for j in os.listdir(primary_path + i):
                kc = torch.from_numpy(np.loadtxt(primary_path + i + "/" + j + "/groundtruth.txt")).type(torch.LongTensor)
                label = kc[:,0]
                c = kc[:,1:]
                data = np.loadtxt(primary_path + i + "/" + j + "/skeletal_sequence.txt")
                d = torch.LongTensor([[k * m, k * m + ws] for k in range((len(data) - ws) // m)])
                cn = [set(range(int(c[k,0]), int(c[k,1]))) for k in range(len(c))]
                dn = [set(range(int(d[k,0]), int(d[k,1]))) for k in range(len(d))]
                vndn = [[np.round(len(cn[k].intersection(dn[df]))/ws * 100) 
                        for df in range(len(dn))] for k in range(len(cn))]
                cd = label[torch.tensor(vndn).max(0)[1]]
                for k in range(len(d)):
                    makedir_path(path + "/" + i + "/" + str(int(cd[k])))
                    np.savetxt(path + "/" + i + "/" + str(int(cd[k])) + "/" + j + "_" + str(int(d[k,0]))
                        + "_" + str(int(d[k,1])) + ".txt" , data[d[k,0]:d[k,1]], fmt="%.6f")
            Nd = len(os.listdir(path + "/" + i))
            limit = {"train": 10000 // Nd, "test": 6000 // Nd}
            for lab in os.listdir(path + "/" + i):
                c0 = os.listdir(path + "/" + i + "/" + lab)
                if (i == "train" and len(c0) > limit[i]) or (i == "test" and len(c0) > limit[i]):
                    for ii in random.sample(c0, len(c0) - limit[i]):
                        os.remove(path + "/" + i + "/" + lab + "/" + ii)