import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize(examples, tokenizer):
    return tokenizer(examples, truncation=True, padding='max_length', max_length=512)

def getMajorityVote(sub, correctSet, tokenizer, trainer):
    dset = []
    for cSub in correctSet:
        dset.append([sub, cSub])
        
    tk_sub = tokenize(dset, tokenizer)
    sub_dataset = Dataset(tk_sub, [0]*len(dset))
    pred = trainer.predict(sub_dataset)
    preds = np.argmax(pred.predictions, axis=1).tolist()
    return int(preds.count(1) / len(preds) >= 0.50)

def getMajorityVoteGrade(sub, correctSet, tokenizer, trainer):
    dset = []
    for cSub in correctSet:
        dset.append([sub, cSub])
        
    tk_sub = tokenize(dset, tokenizer)
    sub_dataset = Dataset(tk_sub, [0]*len(dset))
    pred = trainer.predict(sub_dataset)
    preds = np.array(pred.predictions)[:,1]
    return np.mean(preds)