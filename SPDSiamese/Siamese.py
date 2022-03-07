import torch
import random
from torch import nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Siamese
## Utils
def dis(x1,x2):
  return (x1-x2).pow(2).sum(-1).sqrt()

## Loss Function
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)#.sqrt()  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() *F.relu(self.margin - (distances + self.eps)))#.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

## Model
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)

## Trainer
def train_siamese(epoch, train_loader, model, loss_fn, optimizer, cuda, N, tot, tot_label, rep, m):
    model.train()
    total_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm_notebook(enumerate(train_loader))
    for batch_idx, x in bar:
        data = x['data']
        target = x['label'].type(torch.LongTensor)
        if not type(data) in (tuple, list):
            data = (data, )
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs, )

        loss_inputs = outputs
        if target is not None:
            targets = (target, )
            loss_inputs += targets
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        total_loss += loss.item()
        total += target.size(0)
        loss.backward()
        optimizer.step()
        bar.set_description('Loss: %.3f |(%d/%d)' % (total_loss / (batch_idx + 1.0), 0, total))        
        if batch_idx == len(train_loader) - 1:
            x = model.get_embedding(tot)
            x_rep = model.get_embedding(rep)
            dist_matrix = dis(x.unsqueeze(1).expand(x.size(0), N, N), x_rep.unsqueeze(0).expand(x.size(0), N, N))
            d = dist_matrix.min(1)[1] - tot_label
            Acc = d[d == 0].size(0)
            bar.set_description('Loss: %.3f|Acc : %.2f%% (%d/%d)' % (total_loss / (batch_idx + 1), 100. * Acc / len(x), Acc, len(train_loader.dataset)))
    return (total_loss / (batch_idx + 1), 100. * Acc / len(x))

def test_siamese(epoch, dataloader_val, model, loss_fn, cuda, N, tot, tot_label, rep, m):
    model.eval()
    test_loss = 0
    total = 0.0
    bar = tqdm_notebook(enumerate(dataloader_val))
    for batch_idx, x in bar:
        data = x['data']
        target = x['label'].type(torch.LongTensor)
        if not type(data) in (tuple, list):
            data = (data, )
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        outputs = model(*data)
        if type(outputs) not in (tuple, list):
            outputs = (outputs, )

        loss_inputs = outputs
        if target is not None:
            targets = (target, )
            loss_inputs += targets
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (
            tuple, list) else loss_outputs
        test_loss += loss.data.item()
        total += target.size(0)
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 0., 0, total))
        if batch_idx == len(dataloader_val) - 1:
            x = model.get_embedding(tot)
            x_rep = model.get_embedding(rep)
            dist_matrix = dis(
                x.unsqueeze(1).expand(x.size(0), N, N),
                x_rep.unsqueeze(0).expand(x.size(0), N, N))
            d = dist_matrix.min(1)[1] - tot_label
            Acc = d[d == 0].size(0)
            bar.set_description('Loss: %.3f|Acc : %.2f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * Acc / len(x), Acc, total))
    return (test_loss / (batch_idx + 1), 100. * Acc / len(x))

## Dataloader
class SiameseDataset(Dataset):
    def __init__(self, Data, N):
        self.train = Data.train
        self.transform = Data.transform
        if self.train:
            self.data = Data.data
            self.target = Data.target
            self.label_to_indices = [torch.where(self.target == i)[0] for i in range(N)]
            x = self.label_to_indices
            pos_pairs = [[x[m][i].item(), x[m][i + 1].item(), 1] for m in range(N) for i in range(-1, len(x[m]) - 1)
                         ] + [[x[m][i].item(), x[m][i - 3].item(), 1] for m in range(N) for i in range(3, len(x[m]) - 1, 3)]
            t = []
            Mm = len(pos_pairs)
            h = Mm // N + 1
            for m in range(N):
                r1 = torch.full((h, ), m).tolist()
                r2 = random.sample([i for j in range(h // (N - 1) + 1) for i in set(range(N)) - {m}], h)
                r3 = random.sample(torch.arange(len(x[m])).tolist() + [random.randint(0, len(x[m]) - 1)
                                                                       for i in range(h - len(x[m]))], h)
                r4 = [random.randint(0, len(x[n]) - 1) for n in r2]
                t += torch.tensor([r1, r2, r3, r4]).T.tolist()
            neg_pairs = [[x[m][i], x[n][j], 0] for m, n, i, j in t]
            self.pairs = random.sample(pos_pairs + neg_pairs, len(pos_pairs + neg_pairs))

        else:
            random_state = np.random.RandomState(29)
            self.data = Data.data
            self.target = Data.target
            self.label_to_indices = {i: torch.where(self.target == i)[0] for i in range(N)}
            x = self.label_to_indices
            pos_pairs = [[i, random_state.choice(self.label_to_indices[self.target[i].item()]), 1] 
                         for i in range(0, len(self.data), 2)]
            t = []
            Mm = len(pos_pairs)
            h = Mm // N + 1
            for m in range(N):
                r1 = torch.full((h, ), m).tolist()
                r2 = random.sample([i for j in range(h // (N - 1) + 1) for i in set(range(N)) - {m}], h)
                r3 = random.sample(torch.arange(len(x[m])).tolist() + [random.randint(0, len(x[m]) - 1)
                                                                       for i in range(h - len(x[m]))], h)
                r4 = [random.randint(0, len(x[n]) - 1) for n in r2]
                t += torch.tensor([r1, r2, r3, r4]).T.tolist()
            neg_pairs = [[x[m][i], x[n][j], 0] for m, n, i, j in t]
            self.pairs = random.sample(pos_pairs + neg_pairs, len(pos_pairs + neg_pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s1 = torch.tensor(self.data[self.pairs[idx][0]], requires_grad=False)
        s2 = torch.tensor(self.data[self.pairs[idx][1]], requires_grad=False)
        i1 = self.pairs[idx][2]
        return {'data': (s1, s2), 'label': i1}
