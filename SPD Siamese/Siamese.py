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
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix
class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError
class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs

class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError
class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None
def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None
def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None
class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)
def HardestNegativeTripletSelector(margin, cpu=False): 
  return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=hardest_negative,cpu=cpu)

def RandomNegativeTripletSelector(margin, cpu=False): 
  return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=cpu)
def SemihardNegativeTripletSelector(margin, cpu=False): 
  return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=lambda x: semihard_negative(x, margin),cpu=cpu)
## Metric
class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError
class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'
class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
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
            bar.set_description('Loss: %.3f|Acc : %.2f%% (%d/%d)' %
                                (total_loss / (batch_idx + 1), 100. * Acc / len(x), Acc, len(train_loader.dataset)))
            
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

        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (test_loss / (batch_idx + 1), 0., 0, total))
        if batch_idx == len(dataloader_val) - 1:
            x = model.get_embedding(tot)
            x_rep = model.get_embedding(rep)
            dist_matrix = dis(
                x.unsqueeze(1).expand(x.size(0), N, N),
                x_rep.unsqueeze(0).expand(x.size(0), N, N))
            d = dist_matrix.min(1)[1] - tot_label
            Acc = d[d == 0].size(0)
            bar.set_description(
                'Loss: %.3f|Acc : %.2f%% (%d/%d)' %
                (test_loss / (batch_idx + 1), 100. * Acc / len(x), Acc, total))
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
