import os
import pickle
import torch
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd


def _collate(datalist):
    if len(datalist) == 1:
        datalist = datalist * 2

    def padding(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), max(length)).long()
        for i in range(len(data)):
            encoding[i,:length[i]] = data[i]
        return encoding

    datalist = list(zip(*datalist))
    gene,compound, protein, affinity = datalist
    compound_batch = torch.Tensor(compound)
    protein_batch = padding(protein)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return gene,(compound_batch, protein_batch), affinity_batch

class DeepConvDTIData(object):

    def register_init_feature(self):
        self.seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
        self.seq_dic = {w: i+1 for i,w in enumerate(self.seq_rdic)}
        self.max_seq_len = 1200


    def get_compound(self, smiles):
        #给Smiles，转成Token
     
        mol = Chem.MolFromSmiles(smiles)
        compound = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)

        return compound
        
    def get_protein(self, seq):
        #给protein seq，转成Token
        protein = torch.LongTensor([self.seq_dic[i] for i in seq[:self.max_seq_len]])
        
        return protein


class _Dataset(Dataset, DeepConvDTIData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, df,pro_dir):
        
        self.df = df
        self.ids = self.df.index.values
        df_seq = pd.read_excel(pro_dir)
        self.pro2seq = df_seq.set_index('gene')['seq'].to_dict()
        # load data
        self.register_init_feature()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # get affinity
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        smi = row['smiles']
        gene = row['gene']
        seq = self.pro2seq[gene]
        label = row['max']

        compound = self.get_compound(smi)
        protein = self.get_protein(seq)

        return gene,compound,protein,label

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