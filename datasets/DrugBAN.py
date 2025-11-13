# adapters/fuschem_adapter.py
import torch
import dgl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from rdkit import Chem
import logging
import os
import numpy as np
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

CHARPROTSET = {
    "A": 1,    "C": 2,    "B": 3,    "E": 4,    "D": 5,    "G": 6,
    "F": 7,    "I": 8,    "H": 9,    "K": 10,    "M": 11,    "L": 12,
    "O": 13,    "N": 14,    "Q": 15,    "P": 16,    "S": 17,    "R": 18,
    "U": 19,    "T": 20,    "W": 21,    "V": 22,    "Y": 23,    "X": 24,
    "Z": 25,}

CHARPROTLEN = 25

def integer_label_protein(sequence, max_length=1200):

    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


class _Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, pro_dir,max_drug_nodes=150):
        self.df = df

        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        df_seq = pd.read_excel(pro_dir)
        self.pro2seq = df_seq.set_index('gene')['seq'].to_dict()


    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        smi = row['smiles']
        gene = row['gene']
        label = row['max']
        
        # chem graph
        smi_graph = self.fc(smiles=smi, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = smi_graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        smi_graph.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        smi_graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        smi_graph = smi_graph.add_self_loop()

        # protein
        seq = self.pro2seq[gene]
        pro_seq = integer_label_protein(seq)
        
        return gene,smi_graph, pro_seq, label
    



def _collate(batch):
    gene,smi_graph, pro_seq, label = zip(*batch)
    smi_graph = dgl.batch(smi_graph)
    
    label = torch.tensor(label).unsqueeze(1).float()
    return gene, smi_graph, torch.tensor(np.array(pro_seq)), label

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