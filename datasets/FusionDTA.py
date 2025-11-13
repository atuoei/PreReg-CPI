# adapters/fuschem_adapter.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from rdkit import Chem
from functools import lru_cache
import os

class SmilesEncoder:
    def __init__(self):
        chars = b'#%/\\@)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty'
        self.table = torch.full((256,), 255, dtype=torch.uint8)
        self.table[torch.tensor(list(chars))] = torch.arange(len(chars), dtype=torch.uint8)

    def encode(self, s: str) -> torch.LongTensor:
        b = torch.tensor(list(s.encode('utf-8')), dtype=torch.uint8)
        return self.table[b].to(torch.long)

class _Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, esm_dir: str):
        self.df = df.reset_index(drop=True)
        self.encoder = SmilesEncoder()
        self.esm_dir = esm_dir

    @lru_cache(maxsize=4096)  # 缓存常用基因向量
    def _load_esm(self, gene: str):
        path = os.path.join(self.esm_dir, f"{gene}.pt")
        return torch.load(path, map_location="cpu")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        smiles = self.encoder.encode(r["smiles"])
        prot_esm = self._load_esm(r["gene"])
        return r["gene"], smiles, prot_esm, float(r["max"])

def _collate(batch):
    genes, smiles_list, prot_list, labels = zip(*batch)
    prot = pad_sequence(prot_list, batch_first=True, padding_value=0.0)
    # 保留list给模型（与原代码一致）；如需pad，可在此处理
    labels = torch.tensor(labels, dtype=torch.float32)
    return list(genes), list(smiles_list), prot, labels

class Modeldataset:
    def __init__(self, df_train, df_val, df_test, esm_dir='../Features/proallfeat', batch_size=128, num_workers=4):
        self.train_ds = _Dataset(df_train, esm_dir)
        self.val_ds   = _Dataset(df_val,   esm_dir)
        self.test_ds  = _Dataset(df_test,  esm_dir)
        self.bs = batch_size; self.nw = num_workers

    def dataloaders(self):
        train = DataLoader(self.train_ds, batch_size=self.bs, shuffle=True,
                           num_workers=self.nw, collate_fn=_collate, drop_last=False)
        val   = DataLoader(self.val_ds, batch_size=self.bs, shuffle=False,
                           num_workers=0, collate_fn=_collate)
        test  = DataLoader(self.test_ds, batch_size=self.bs, shuffle=False,
                           num_workers=0, collate_fn=_collate)
        return train, val, test


