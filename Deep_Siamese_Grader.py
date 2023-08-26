import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings1, encodings2, labels):
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        sub1 = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        sub2 = {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        
        return sub1, sub2, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

def tokenize(examples, tokenizer):
    return [tokenizer(np.array(examples)[:,i].tolist(), truncation=True, padding='max_length', max_length=512) for i in [0, 1]]

class SiameseNetwork(torch.nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.model = encoder

    def forward_once(self, x):
        output = self.model(input_ids=x[0], attention_mask=x[1])[0]
        output = output.view(output.size(0), -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(0.5 * (label) * torch.pow(euclidean_distance, 2) +
                                      0.5 * (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def testing(dataloader, model):
    preds = []
    labels = []
    for i, data in enumerate(dataloader, 0):
        sub1, attn1 = data[0].values()
        sub2, attn2 = data[1].values()
        label = data[2]
        sub1, attn1 = Variable(sub1).cuda(), Variable(attn1).cuda()
        sub2, attn2 = Variable(sub2).cuda(), Variable(attn2).cuda()
        output1, output2 = model((sub1, attn1), (sub2, attn2))
        euclidean_distance = F.pairwise_distance(output1, output2)
        preds.extend(euclidean_distance.tolist())
        labels.extend(label.tolist())
    return preds, labels

def getMajorityVote(sub, correctSet, tokenizer, batch_size, model):
    dset = []
    for cSub in correctSet:
        dset.append([sub, cSub])
        
    tk_sub = tokenize(dset, tokenizer)
    sub_dataset = Dataset(tk_sub[0], tk_sub[1], [0]*len(dset))
    dataloader = DataLoader(sub_dataset, batch_size=batch_size)
    preds, _ = testing(dataloader, model)
    preds = (1 / np.exp(preds) > 0.5).tolist()
    return int(preds.count(1) / len(preds) >= 0.50)

def getMajorityVoteGrade(sub, correctSet, tokenizer, batch_size, model):
    dset = []
    for cSub in correctSet:
        dset.append([sub, cSub])
        
    tk_sub = tokenize(dset, tokenizer)
    sub_dataset = Dataset(tk_sub[0], tk_sub[1], [0]*len(dset))
    dataloader = DataLoader(sub_dataset, batch_size=batch_size)
    preds, labels = testing(dataloader, model)
    return np.mean(preds)
