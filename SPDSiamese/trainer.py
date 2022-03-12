from tqdm import tqdm_notebook
import torch
# Trainer 
# Training
def train(epoch, model, optimizer, criterion, dataloader, use_cuda):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm_notebook(enumerate(dataloader))
    for batch_idx, sample_batched in bar:
        inputs = torch.tensor(sample_batched['data'], requires_grad=True)
        targets = sample_batched['label'].type(torch.LongTensor)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        if type(outputs) not in (tuple, list):
            loss = criterion(outputs, targets)
        else:
            outputs = outputs[0]
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                        (train_loss / (batch_idx + 1.0), 100. * correct / total, correct, total))
    return (train_loss / (batch_idx + 1), 100. * correct / total)

def test(epoch, model, optimizer, criterion, dataloader_test, use_cuda):
    model.eval()
    test_loss = 0
    correct = 0.0
    total = 0.0
    bar = tqdm_notebook(enumerate(dataloader_test))
    for batch_idx, sample_batched in bar:
        inputs = torch.tensor(sample_batched['data'], requires_grad=True)
        targets = sample_batched['label'].type(torch.LongTensor)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.type(torch.LongTensor).cuda()
        outputs = model(inputs)
        if type(outputs) not in (tuple, list):
            loss = criterion(outputs, targets)
        else:
            outputs = outputs[0]
            loss = criterion(outputs, targets)
        loss = criterion(outputs, targets.type(torch.LongTensor))
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().data.item()
        bar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (test_loss /(batch_idx + 1.0), 100. * correct / total, correct, total))
    return (test_loss / (batch_idx + 1), 100. * correct / total)
