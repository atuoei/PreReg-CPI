# adapters/fuschem_adapter.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1200):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X



class _Dataset(Dataset):
    def __init__(self, df,pro_dir):
        self.df = df
        self.ids = self.df.index.values
        df_seq = pd.read_excel(pro_dir)
        self.pro2seq = df_seq.set_index('gene')['seq'].to_dict()



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        smiles = row['smiles']
        gene = row['gene']
        label = row['max']
        pro_seq = self.pro2seq[gene]        
        
        return gene,smiles, pro_seq, label


def _collate(batch,max_d=150,max_p=1000):

    N = len(batch)
    gene,_, _, label = zip(*batch)
    compound_new = torch.zeros((N, max_d), dtype=torch.long)
    protein_new = torch.zeros((N, max_p), dtype=torch.long)
    label = torch.tensor(label).unsqueeze(1).float()
    for i,pair in enumerate(batch):
       
        compoundstr, proteinstr = pair[-3], pair[-2]
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET,max_d))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET,max_p))
        protein_new[i] = proteinint
    
    return gene,compound_new, protein_new, label

class Modeldataset:
    def __init__(self, df_train, df_val, df_test, pro_dir='../Features/targets.csv', batch_size=128, num_workers=4):
        self.train_ds = _Dataset(df_train, pro_dir)
        self.val_ds   = _Dataset(df_val,   pro_dir)
        self.test_ds  = _Dataset(df_test,  pro_dir)
        self.bs = batch_size; self.nw = num_workers

    def dataloaders(self):
        train = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,
                           num_workers=self.nw, collate_fn=_collate, drop_last=False)
        val   = DataLoader(self.val_ds, batch_size=self.bs, shuffle=False,
                           num_workers=0, collate_fn=_collate)
        test  = DataLoader(self.test_ds, batch_size=self.bs, shuffle=False,
                           num_workers=0, collate_fn=_collate)
        return train, val, test