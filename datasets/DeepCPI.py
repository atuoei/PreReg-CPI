import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset,DataLoader


# 自定义collate函数
def _collate(datalist):
    if len(datalist) == 1:
        datalist = datalist * 2
        
    datalist = list(zip(*datalist))
    target,compound, protein, affinity = datalist
    compound_batch = torch.Tensor(compound)
    # compound_batch = compound_batch.squeeze(1)
    protein_batch = torch.Tensor(protein)
    label_batch = torch.Tensor(affinity).reshape(-1, 1)
    return target,compound_batch, protein_batch, label_batch


class _Dataset(Dataset):
    def __init__(self, df,pro_dir):
        self.df = df
        self.ids = self.df.index.values
        self.df_prow2v = pd.read_csv(pro_dir,index_col=0)
    def __len__(self):
        return len(self.df)
    
    def smiles_to_Morgan(self,smi, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smi)
        finger = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True))
        return finger
      
    def gene2feat(self,gene):
        feat = self.df_prow2v.loc[gene,:].values
        return feat

    def __getitem__(self, idx):
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        smi = row['smiles']
        gene = row['gene']
        label = row['max']
        Morgan = self.smiles_to_Morgan(smi)
        prot_esm = self.gene2feat(gene)        
        
        return gene,Morgan, prot_esm, label

class Modeldataset:
    def __init__(self, df_train, df_val, df_test, pro_dir='../Features/esm_150.csv', batch_size=128, num_workers=4):
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