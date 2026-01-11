import sys
import os
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from .utils import *

class CustomDataset(Dataset):
    def __init__(self, in_file, chroms=[], idx_ch=1, idx_X=-2, idx_y=-1, alphabet=['N', 'A', 'C', 'G', 'T']):
        self.X = []
        self.y = []
        self.alphabet = alphabet
        with open(in_file) as inFile:
            head = inFile.readline()
            for line in inFile:
                line = line.strip()
                fields = line.split('\t')
                ch = fields[idx_ch]
                if chroms:
                    if ch in chroms:
                        self.X.append(fields[idx_X])
                        self.y.append(fields[idx_y])
                else:
                    self.X.append(fields[idx_X])
                    self.y.append(fields[idx_y])
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        X = [self.alphabet.index(x) if x in self.alphabet else 0 for x in X]
        y = self.OneHot(list(y), alphabet=['0', '1', '2'])
        X = torch.tensor(X, dtype=torch.int32)
        y = torch.tensor(y, dtype=torch.float32)
        return(X, y)

    def OneHot(self, L, alphabet=['A', 'C', 'G', 'T']):
        cat = list(np.array(sorted(alphabet)).reshape(1, -1))
        oe = OneHotEncoder(categories=cat, handle_unknown='ignore')
        s = oe.fit_transform(np.array(L).reshape(-1, 1)).toarray().transpose(1, 0)
        return(s)

    def split_and_save_dataset(self, out_file, batch_size=32, shuffle=True, num_workers=0):
        ds = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        torch.save(ds, out_file)
