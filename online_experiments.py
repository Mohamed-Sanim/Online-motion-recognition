from SPDSiamese.model import ST_TS_SPDC, Net
from SPDSiamese.Siamese import * 
import argparse
import pickle
import shutil, os
import torch
from torch import nn
import numpy as np
from prettytable import PrettyTable
import collections
import warnings
import operator
import vg
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Online recognition exectution")
parser.add_argument(
    "--path", 
    type = str, 
    default = "", 
    help = "Path to either the folder containing the skeletal data", )

parser.add_argument(
    "--dataset", 
    type = str, 
    default = "OAD", 
    choices = ["ODHG", "OAD", "UOW", "InHard"], 
    help = "the proposed skeletal datasets", 
)


parser.add_argument(
    "--te", 
    type = int, 
    default = 3, 
    help = "number of tests in the verification process", 
)

parser.add_argument(
    "--threshold", 
    type = float, 
    default = 0.6, 
    help = "threshold of Jaccard index. An action is detected if the Jaccard index \
    between the groundtruth interval and the predicted interval exceeds this threshold", 
)

parser.set_defaults(feature = False)
args = parser.parse_args()
path = args.path + "/" + args.dataset 

rate = {"ODHG": 30, "OAD": 8, "UOW": 20, "InHard": 30}
#Partitioning
parts = dict()
parts["InHard"] = [[12, 11, 10, 0, 1, 2, 3], [12, 11, 10, 0, 4, 5, 6], [12, 11, 10, 13, 14, 15, 16], [12, 11, 10, 17, 18, 19, 20]]
parts["ODHG"] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
parts["OAD"] = [[3, 2, 20, 4, 5, 6, 7, 21, 22], [3, 2, 20, 1, 0, 12, 13, 14, 15], [3, 2, 20, 1, 0, 16, 17, 18, 19], [3, 2, 20, 8, 9, 10, 11, 23, 24]]
parts["UOW"] = [[3, 2, 20, 4, 5, 6, 7, 21, 22], [3, 2, 20, 1, 0, 12, 13, 14, 15], [3, 2, 20, 1, 0, 16, 17, 18, 19], [3, 2, 20, 8, 9, 10, 11, 23, 24]]
type_detector = {"ODHG" : "Binary", "OAD" : "Binary", "InHard" : "Binary", "UOW" : "General"}

print("Preparing the Classifier...")
Nc = len(os.listdir(path + "/Classifier/train/"))  #number of classes
Classifier_parameters = torch.load(path + "/Classifier/models/parameters.pth")
Classifier_classes = Classifier_parameters["ori_classes"]
model_classify = ST_TS_SPDC(Nc, parts[args.dataset], t0 = Classifier_parameters["t0"], 
    NS = Classifier_parameters["NS"], eps = Classifier_parameters["eps"], 
    vect = True, outs = Classifier_parameters["outs_trans"])
wn_state = dict(model_classify.state_dict())
wn_state['conv.weight'] = dict(torch.load(path + "/Classifier/models/spd_learning.pth"))['conv.weight']
wn_state['spdagg.weight'] = dict(torch.load(path + "/Classifier/models/spd_learning.pth"))['spdagg.weight']
wn_state['fc.weight'] = dict(torch.load(path + "/Classifier/models/siamese_best.pth")['state'])['embedding_net.fc.weight']
wn_state['fc.bias'] = dict(torch.load(path + "/Classifier/models/siamese_best.pth")['state'])['embedding_net.fc.bias']
wn_state = collections.OrderedDict(wn_state)
model_classify.load_state_dict(wn_state)
Classifier_acc = np.round(torch.load(path + "/Classifier/models/siamese_best.pth")['acc'] * 10000)/100
repc = torch.load(path + "/Classifier/models/siamese_best.pth")['rep']
model_c = SiameseNet(Net(Nc, Classifier_parameters["outs_trans"]))
model_c.load_state_dict(torch.load(path + "/Classifier/models/siamese_best.pth")['state'])
x_repc = model_c.get_embedding(repc)

print("Preparing the Detector...")
Nd = len(os.listdir(path + "/Detector/train/")) #number of classes
Detector_parameters = torch.load(path + "/Detector/models/parameters.pth")
Detector_classes = Detector_parameters["ori_classes"]
model_detect = ST_TS_SPDC(Nd, parts[args.dataset], t0 = Detector_parameters["t0"], 
    NS = Detector_parameters["NS"], eps = Detector_parameters["eps"], 
    vect = True, outs = Detector_parameters["outs_trans"])
wn_state = dict(model_detect.state_dict())
wn_state['conv.weight'] = dict(torch.load(path + "/Detector/models/spd_learning.pth"))['conv.weight']
wn_state['spdagg.weight'] = dict(torch.load(path + "/Detector/models/spd_learning.pth"))['spdagg.weight']
wn_state['fc.weight'] = dict(torch.load(path + "/Detector/models/siamese_best.pth")['state'])['embedding_net.fc.weight']
wn_state['fc.bias'] = dict(torch.load(path + "/Detector/models/siamese_best.pth")['state'])['embedding_net.fc.bias']
wn_state = collections.OrderedDict(wn_state)
model_detect.load_state_dict(wn_state)
Detector_acc = np.round(torch.load(path + "/Detector/models/siamese_best.pth")['acc'] * 10000)/100
repd = torch.load(path + "/Detector/models/siamese_best.pth")['rep']
model_d = SiameseNet(Net(Nd, Detector_parameters["outs_trans"]))
model_d.load_state_dict(torch.load(path + "/Detector/models/siamese_best.pth")['state'])
x_repd = model_d.get_embedding(repd)

print("Online recognition running...")
full_path = os.listdir(path + "/Online/test/")
te = args.te
ws = Detector_parameters["ws"]
m = Detector_parameters["m"]
rt = {}
d ={}
vi = {}
indic_start = {}
indic_end = {}
if type_detector[args.dataset] == "Binary":
    for k in full_path:
        d[k] = []
        vi[k] = []
        indic_start[k] = []
        indic_end[k] = []
        h = 0
        data_tot = np.loadtxt(path + "/Online/test/" + k + "/skeletal_sequence.txt")
        for i in range(len(data_tot) // m):
            n = (ws // m) - 1
            if i < n:
                x = data_tot[:m * (i + 1)]
            else:
                x = data_tot[m * (i - n):m * (i + 1)]
            data = torch.from_numpy(x)
            row, col = np.shape(data)
            data = data.T.reshape((1, col, row))
            data = nn.functional.interpolate(data, size=Detector_parameters["interpolation"]).squeeze().T
            data = vg.normalize(data)
            data = data.reshape((1, data.size(0), col // 3, 3, 1))
            data = model_detect(data)[0]
            dist_matrix = dis(data.unsqueeze(1).expand(data.size(0), Nd, Nd), 
                x_repd.unsqueeze(0).expand(data.size(0), Nd, Nd))
            c = int(dist_matrix.min(1)[1])
            hs = 1 if (c != 0 and len(vi) % 2 == 0) or (len(indic_start) > 0) else 0
            if len(vi[k]) % 2 == 0 and hs == 1:
                indic_start[k].append(c)
                if len(indic_start[k]) == te:
                    c = np.round(sum(indic_start[k]) / te)
                    if c != 0:
                        vi[k].append(m * (i + 1) - te * m)
                    indic_start[k] = []
            he = 1 if (c == 0 and len(vi[k]) % 2 == 1) or (len(indic_end[k]) > 0) else 0
            if len(vi[k]) % 2 == 1 and he == 1:
                indic_end[k].append(c)
                if len(indic_end[k]) == te:
                    c = np.round(sum(indic_start[k]) / te)
                    if c == 0:
                        vi[k].append(m * (i + 1) - te * m)
                        data_c = torch.from_numpy(data_tot[vi[k][-2]:vi[k][-1]])
                        row, col = np.shape(data_c)
                        data_c = data_c.T.reshape((1, col, row))
                        data_c = nn.functional.interpolate(data_c, size=Classifier_parameters["interpolation"]).squeeze().T
                        data_c = vg.normalize(data_c)
                        data_c = data_c.reshape((1, data_c.size(0), col // 3, 3, 1))
                        data_c = model_classify(data_c)[0]
                        dist_matrix = dis(data_c.unsqueeze(1).expand(data_c.size(0), Nc, Nc),
                                          x_repc.unsqueeze(0).expand(data_c.size(0), Nc, Nc))
                        cc = int(dist_matrix.min(1)[1])
                        d[k].append(Classifier_classes[cc])
                    indic_end[k] = []
        if len(vi[k]) % 2 == 1 and i == len(data_tot) // m - 1:
            vi[k].append(len(data_tot))
            data_c = torch.from_numpy(data_tot[vi[k][-2]:vi[k][-1]])
            row, col = np.shape(data_c)
            data_c = data_c.T.reshape((1, col, row))
            data_c = nn.functional.interpolate(data_c, size=Classifier_parameters["interpolation"]).squeeze().T
            data_c = vg.normalize(data_c)
            data_c = data_c.reshape((1, data_c.size(0), col // 3, 3, 1))
            data_c = model_classify(data_c)[0]
            dist_matrix = dis( data_c.unsqueeze(1).expand(data_c.size(0), Nc, Nc), 
                              x_repc.unsqueeze(0).expand(data_c.size(0), Nc, Nc))
            cc = int(dist_matrix.min(1)[1])
            d[k].append(Classifier_classes[cc])
        rt[k] = torch.tensor([d[k],vi[k][::2],vi[k][1::2]]).T
else:
    def max_iter(L):
        stats = {s: L.count(s) for s in set(L)}
        return max(stats.items(), key=operator.itemgetter(1))[0]
    for k  in full_path:
        d[k] = []
        vi[k] = [0]
        indic_start[k] = []
        cl_K = []
        h = 0
        data_tot = np.loadtxt(path + "/Online/test/" + k + "/skeletal_sequence.txt")
        for i in range(len(data_tot) // m):
            n = (ws // m) - 1
            if i < n:
                x = data_tot[:m * (i + 1)]
            else:
                x = data_tot[m * (i - n):m * (i + 1)]
            data = torch.from_numpy(x)
            row, col = np.shape(data)
            data1 = data.T.reshape((1, col, row))
            data = nn.functional.interpolate(data1, size=Detector_parameters["interpolation"]).squeeze().T
            data = data.reshape((1, data.size(0), col // 3, 3, 1))
            data = model_detect(data)[0]
            dist_matrix = dis(data.unsqueeze(1).expand(data.size(0), Nd, Nd),
                              x_repd.unsqueeze(0).expand(data.size(0), Nd, Nd))
            c = int(dist_matrix.min(1)[1])
            cl_K.append(c)
            c = max_iter(cl_K[-te:]) if len(cl_K)>2 else max_iter(cl_K)
            cn = max_iter(cl_K[-(te + 1):-1]) if len(cl_K)>1 else -1
            hs = 1 if ((c != cn) or (len(indic_start[k]) > 0)) and len(cl_K)>1 else 0
            if (hs == 1):
                indic_start[k].append(c)
                if len(indic_start[k]) == ws:
                    c = max_iter(indic_start[k])
                    if  (len(d[k])>0 and c != d[k][-1]) or c!= -1:
                        vi[k].append(m * (i + 1) - ws * m)
                        datac = torch.from_numpy(data_tot[vi[k][-2]:vi[k][-1]])
                        row, col = np.shape(datac)
                        datac1 = datac.T.reshape((1, col, row))
                        datac = nn.functional.interpolate(datac1, size=Classifier_parameters["interpolation"]).squeeze().T
                        datac = datac.reshape((1, datac.size(0), col // 3, 3, 1))
                        datac = model_classify(datac)[0]
                        dist_matrix = dis(datac.unsqueeze(1).expand(datac.size(0), Nc, Nc),
                                          x_repc.unsqueeze(0).expand(datac.size(0), Nc, Nc))
                        cc = int(dist_matrix.min(1)[1])
                        d[k].append(cc)
                    indic_start[k] = []
            if (i == len(data_tot) // m -1) and (vi[k][-1]!= len(data_tot) // m - 1):
                vi[k].append(len(data_tot))
                datac = torch.from_numpy(data_tot[vi[k][-2]:])
                row, col = np.shape(datac)
                datac1 = datac.T.reshape((1, col, row))
                datac = nn.functional.interpolate(datac1, size=Classifier_parameters["interpolation"]).squeeze().T
                datac = datac.reshape((1, datac.size(0), col // 3, 3, 1))
                datac = model_classify(datac)[0]
                dist_matrix = dis(datac.unsqueeze(1).expand(datac.size(0), Nc, Nc),
                                  x_repc.unsqueeze(0).expand(datac.size(0), Nc, Nc))
                cc = int(dist_matrix.min(1)[1])
                d[k].append(cc)
            ii=0
            while ii<len(d[k])-1:
                if d[k][ii] == d[k][ii+1]:
                    del d[k][ii+1],vi[k][ii+1]
                else:
                    ii+=1
            rt[k] = torch.cat((torch.tensor([d[k]]).T, torch.tensor([vi[k][:-1]]).T + 1, 
                torch.tensor([vi[k][1:]]).T), 1)

ll = ['Classifier accuracy(%)', 'Detector accuracy(%)', 'Detection', 'Correct Detection', 'SL', 'EL', 'Global Precision', 'Global recall',
      '  Gloabl F1-score  ']
myTable = PrettyTable()
myTable.add_column(" ", ll)

for nnn in range(2):
    rtt = {}
    for h in rt:
        i = 0
        rt[h] = rt[h].tolist()
        rtt[h] = []
        while i <len(rt[h])-1:
            if rt[h][i][0] == rt[h][i+1][0] and rt[h][i+1][1] - rt[h][i][2] < 2 * rate[args.dataset] // 3:
                rtt[h].append([rt[h][i][0],rt[h][i][1],rt[h][i+1][2]])
                i+=2
                if i == len(rt[h])-1:
                    rtt[h].append(rt[h][i])
            else:
                rtt[h].append(rt[h][i])
                i+=1
                if i == len(rt[h])-1:
                    rtt[h].append(rt[h][i])
        rtt[h] = torch.tensor(rtt[h])
    rt = rtt
for k in rt:
    np.savetxt(path + "/Online/test/" + k + "/predicted_segments.txt", rt[k], fmt="%i")
gt = det_jac = label = SL = EL = gt_labels = labels = gt_inte = inte = torch.tensor([])
for i in full_path:
    z = torch.from_numpy(np.loadtxt(path + "/Online/test/" + k + "/groundtruth.txt"))
    vge, vb, ve = z.type(torch.LongTensor).T
    d, vdb, vde = rt[i].type(torch.LongTensor).T
    vn = [set(range(vb[k],ve[k]+1)) for k in range(len(z))]
    dn = [set(range(vdb[k],vde[k]+1)) for k in range(len(vdb))]
    vndn = [[np.round(len(vn[k].intersection(dn[df]))/len(vn[k].union(dn[df]))*100) for df in range(len(dn))]
            for k in range(len(z))]
    a,b = torch.tensor(vndn).type(torch.LongTensor).max(1)
    det_jac = torch.cat((det_jac,a))
    gt_labels = torch.cat((gt_labels, vge))
    labels = torch.cat((labels,d[b]))
    gt_inte = torch.cat((gt_inte,z[:,1:]))
    inte = torch.cat((inte, rt[i][:,1:][b]))
theta = args.threshold
det = len(det_jac[det_jac >= theta]) / len(det_jac) 
det_cor = len(torch.where(labels[det_jac >= theta] - gt_labels[det_jac >= theta] ==0)[0])/len(det_jac)
gt = {i: len(gt_labels[gt_labels == i]) for i in range(Nc)}
gd = {i: len(torch.where(labels[det_jac >= theta] == i)[0]) for i in range(1,Nc)}
gdc = {i: len(torch.where(labels[det_jac >= theta][torch.where(gt_labels[det_jac >= theta] == i)[0]] == i)[0])
        for i in range(1,Nc)}
SL= (-(inte[:,0][det_jac>=theta] - gt_inte[:,0][det_jac >= theta]).abs() /
      (gt_inte[:,1][det_jac>=theta] - gt_inte[:,0][det_jac >= theta])).exp().sum() / len(gt_inte)
EL= (-(inte[:,1][det_jac>=theta] - gt_inte[:,1][det_jac >= theta]).abs() /
      (gt_inte[:,1][det_jac>=theta] - gt_inte[:,0][det_jac >= theta])).exp().sum() / len(gt_inte)
r = np.sum(list(gdc.values()))/np.sum(list(gd.values()))
rr = np.sum(list(gdc.values()))/np.sum(list(gt.values()))
myTable.add_column('ws=' + str(ws),
            np.round(1000* np.array([Classifier_acc, Detector_acc,det, det_cor,float(SL), float(EL), r, rr, 2*r*rr/(r+rr) ]))/1000)
print(myTable)