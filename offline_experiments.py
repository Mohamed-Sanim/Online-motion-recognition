from SPDSiamese.model import ST_TS_SPDC, Net
from SPDSiamese.Dataloader import Dataset_learning, Dataset_classification
from SPDSiamese.trainer import train, test
from SPDSiamese.Siamese import *
from SPDSiamese.optimizer import *
import argparse
import pickle
import shutil, os
import torch
from torch import nn
import numpy as np

parser = argparse.ArgumentParser(description="Online recognition parameters and datasets")
parser.add_argument(
    "--path",
    type=str,
    default="",
    help="Path to either the folder containing the skeletal data",)

parser.add_argument(
    "--dataset",
    type=str,
    default="OAD",
    choices=["ODHG", "OAD", "UOW", "InHard"],
    help="the proposed skeletal datasets",
)

parser.add_argument(
    "--execution",
    type=str,
    default="Classifier",
    choices=["Classifier", "Detector"],
    help="type of the execution",
)

parser.add_argument(
    "--download",
    type=str,
    default="Yes",
    choices=["No", "Yes"],
    help="if the dataset is not already downloaded on your PC, you put "No" in order to install it.",
)

parser.add_argument(
    "--interpolation",
    type=int,
    default=500,
    help="The skeletal sequences will be normalized to the proposed number of frames",
)


parser.add_argument(
    "--t0",
    type=int,
    default=1,
    help="t0: time interval of the window in spatial-temporal studies",
)

parser.add_argument(
    "--NS",
    type=int,
    default=15,
    help="NS: number of secondary subsequences in each primary subsequnece during the temporal-spatial studies",
)

parser.add_argument(
    "--eps",
    type=float,
    default=10**(-4),
    help="eps: threshold of the ReEig layer",
)

parser.add_argument(
    "--outs_trans",
    type=int,
    default=200,
    help="outs_trans: parameter of the transformation of the SPD matrix size in SPD Aggregation layer",
)

parser.add_argument(
    "--lr",
    type=float,
    default=7 * (10**(-4)),
    help="lr: optimizer learning rate",
)

parser.add_argument(
    "--margin",
    type=float,
    default=7.0,
    help="margin of the contrastive loss function of the Siamese network",
)

parser.add_argument(
    "--m",
    type=int,
    help="m : refresh rate of the detector",
)

parser.add_argument(
    "--ws",
    type=int,
    help="ws : window size of the detector",
)

parser.add_argument(
    "--epoch_learning",
    type=int,
    default = 10,
    help="number of epoch needed for SPD learning",
)

parser.add_argument(
    "--epoch_classfication",
    type=int,
    default = 100,
    help="number of epoch needed for SPD classification",
)

parser.set_defaults(feature=False)

args = parser.parse_args()
path = ["/" + i for i in args.path.split("/")] + "/" + args.dataset + "/" + args.exection

###       Loading dataset      ###
print("Login dataset ...")
transformed_dataset = Dataset_learning(path, args.interpolation, train=True)
dataloader = DataLoader(transformed_dataset, batch_size=30, shuffle=True, num_workers=0)
transformed_dataset_test = Dataset_learning(path, args.interpolation, train=False)
dataloader_test = DataLoader(transformed_dataset_test, batch_size=30, shuffle=False, num_workers=0)

use_cuda = False
#Partitioning
parts = dict()
parts["InHard"] = [[12, 11, 10, 0, 1, 2, 3], [12, 11, 10, 0, 4, 5, 6], [12, 11, 10, 13, 14, 15, 16], [12, 11, 10, 17, 18, 19, 20]]
parts["ODHG"] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
parts["OAD"] = [[3,2,20,4,5,6,7,21,22], [3,2,20,1,0,12,13,14,15], [3,2,20,1,0,16,17,18,19], [3,2,20,8,9,10,11,23,24]]
parts["UOW"] = [[3,2,20,4,5,6,7,21,22], [3,2,20,1,0,12,13,14,15], [3,2,20,1,0,16,17,18,19], [3,2,20,8,9,10,11,23,24]]

N = max([int(i) for i in os.listdir(path)]) # Number of classes

model = ST_TS_SPDC(N, parts[args.dataset], t0 = args.t0, NS=args.NS, eps=args.eps, vect=True, outs = args.outs_trans)
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = StiefelMetaOptimizer(optimizer)

print("-----------------------------------\nSPD Matrices learning...\n-----------------------------------")
start_epoch = 1
for epoch in range(start_epoch, start_epoch + args.learning_epochs):
    train_loss, train_acc = train(epoch, model, optimizer, criterion, dataloader)
    test_loss, test_acc = test(epoch, model, optimizer, criterion, dataloader_test)

os.mkdir(path + "/learnt_spd_matrices/")
os.mkdir(path + "/learnt_spd_matrices/train/")
os.mkdir(path + "/learnt_spd_matrices/test/")
for i in range(N + 1):
  os.mkdir(path + "/learnt_spd_matrices/train/" + str(i))
  os.mkdir(path + "/learnt_spd_matrices/test/" + str(i))
print("Loading SPD matrices...")
for i in range(len(dataloader.dataset)):
  y = model(dataloader.dataset[i]['data'].unsqueeze(0))[1].squeeze()
  l = dataloader.dataset[i]['label']
  np.savetxt(path + "/learnt_spd_matrices/train/" + str(int(l)) + "/" + str(i) + ".txt", y.cpu().detach().numpy())

for i in range(len(dataloader_test.dataset)):
  y = model(dataloader_test.dataset[i]['data'].unsqueeze(0))[1].squeeze()
  l = dataloader_test.dataset[i]['label']
  np.savetxt(path + "/learnt_spd_matrices/test/" + str(int(l)) + "/" + str(i) + ".txt", y.cpu().detach().numpy())

print("Loading Siamese dataset...")
transformed_dataset = Dataset_classification(path + "/learnt_spd_matrices")
train_loader = DataLoader(transformed_dataset, batch_size=30, shuffle=True, num_workers=0)

transformed_dataset_val = Dataset_classification(path + "/learnt_spd_matrices", train=False)
test_loader = DataLoader(transformed_dataset_val, batch_size=30, shuffle=False, num_workers=0)

siamese_transformed_dataset = SiameseDataset(Dataset_classification(path + "/learnt_spd_matrices"), N)
siamese_train_loader = DataLoader(siamese_transformed_dataset, batch_size=30,shuffle=True, num_workers=0)

siamese_transformed_dataset_val = SiameseDataset(Dataset_classification(path + "/learnt_spd_matrices", train=False), N)
siamese_test_loader = DataLoader(siamese_transformed_dataset_val, batch_size=30,shuffle=False, num_workers=0)

tu=[train_loader.dataset[i]['data'].unsqueeze(0) for i in range(len(train_loader.dataset))]
tot=torch.cat(tuple(tu),0)
tu=[train_loader.dataset[i]['label'].unsqueeze(0) for i in range(len(train_loader.dataset))]
tot_label=torch.cat(tuple(tu),0)
tu=[test_loader.dataset[i]['data'].unsqueeze(0) for i in range(len(test_loader.dataset))]
tot_test=torch.cat(tuple(tu),0)
tu=[test_loader.dataset[i]['label'].unsqueeze(0) for i in range(len(test_loader.dataset))]
tot_label_test=torch.cat(tuple(tu),0)

rep=torch.zeros(N, args.outs_trans, args.outs_trans)
for j in range(N):
  i=0
  while int(train_loader.dataset[i]['label'].item())!=j:
    i+=1
  rep[j]=train_loader.dataset[i]['data']
  
margin = args.margin
embedding_net = Net(N)
model = SiameseNet(embedding_net)
cuda = False
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

start_epoch = 1
loss_train,loss_test,acc_train,acc_test = {},{},{},{}
print("-----------------------------------\nSiamese network :training...\n-----------------------------------")
for epoch in range(start_epoch, start_epoch + args.classification_epochs):
    print("\nEpoch ",epoch,":")
    train_loss, train_acc = train_siamese(epoch, siamese_train_loader, model, loss_fn, optimizer, cuda, N, tot, tot_label, rep, margin)
    test_loss, test_acc  = test_siamese(epoch, siamese_test_loader, model, loss_fn, cuda, N, tot_test, tot_label_test, rep, margin)
    loss_train[epoch],loss_test[epoch],acc_train[epoch],acc_test[epoch] = train_loss,test_loss,train_acc,test_acc
    L=[loss_train,loss_test,acc_train,acc_test]
    if test_acc>= max(acc_test.values()):
        torch.save(model.state_dict(), path + "/models/inhard_siamese_best.pth")

print("Choosing best representers ...")
model.load_state_dict(torch.load(path + "/models/inhard_siamese_best.pth"))
if len(tot_test) < 2000:
  x_test=model.get_embedding(tot_test)
else:
  x_test = torch.empty(len(tot_test), N, (args.outs_trans * (args.outs_trans + 1)) // 2)
  for i in range(len(tot_test) // 2000 - 2):
    x_test[2000 * i: 2000 * (i+1)]=model.get_embedding(tot_test[2000 * i: 2000 * (i+1)])
  x_test[2000 * (i + 1):]=model.get_embedding(tot_test[2000 * (i + 1):])

repp = rep.clone()
rep_best = repp
acc = {}
acc_best = {}
for i in range(N):
    acc[i] = {}
    acc_best[i] = []
    c = torch.where(tot_label == i)[0]
    for j in range(len(c)):
        rep_best[i] = tot[c[j]]
        x_rep = model.get_embedding(rep_best)
        dist_matrix = dis(x_test.unsqueeze(1).expand(x_test.size(0), N, N), x_rep.expand(x_test.size(0), N, N))
        d = dist_matrix.min(1)[1] - tot_label_test
        Acc = d[d == 0].size(0)
        acc[i][int(c[j].item())] = Acc / len(tot_label_test)
        l = []
    l = list(acc[i].values())
    l.sort()
    cc = l[-1]
    for k in acc[i]:
        if acc[i][k] == cc:
            acc_best[i].append(k)
    rep_best[i] = tot[acc_best[i][0]]
    
def construct_tree(acc_best, r, m, M):
  for i in range(m, M):
    h = []
    x2 = torch.tensor(acc_best[i]).reshape(len(acc_best[i]), 1).tolist()
    for j in range(len(r)):
        for k in range(len(x2)):
            h.append(r[j] + x2[k])
    r = h
  return r

r = acc_best[0]
r = torch.tensor(r).reshape(len(r), 1).tolist()
m, M = 1, 2
while M < N:
  while len(construct_tree(acc_best, r, m, M)) < 2000 and M < N:
    M += 1
  if len(construct_tree(acc_best, r, m, M)) > 4000:
    M -= 1
  accc = []
  for i in range(len(r)):
      rep_best[:M] = tot[r[i]]
      x_rep = model.get_embedding(rep_best)
      dist_matrix = dis(x_test.unsqueeze(1).expand(x_test.size(0), N, N), x_rep.expand(x_test.size(0), N, N))
      d = dist_matrix.min(1)[1] - tot_label_test
      Acc = d[d == 0].size(0)
      accc.append(Acc / len(d))
  r = r[accc.index(max(accc))]
  r = torch.tensor(r).reshape(1,len(r)).tolist()
  m = M
  M += 1
torch.save({"state":model.state_dict(), "rep":rep_best, "acc": max(accc)}, path + "/models/inhard_siamese_best.pth")
print("The exectution is finalized. You can check the models in ", path, "/model/.")
