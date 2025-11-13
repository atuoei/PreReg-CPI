
from importlib import import_module
import torch
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset,DataLoader

# 定义dataset
class GraphDTAData(object):

    def get_compound(self, smiles):
        mol = self.get_mol(smiles)
        compound = self.mol_to_graph(mol)

        return compound
        
    def get_protein(self, seq):
        protein = torch.LongTensor([self.dict_char_seq[i] for i in seq[:self.max_seq_len]])
        
        return protein

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)


        return mol

    def register_init_feature(self):
        #load intial atom and bond features (i.e., embeddings)        
        seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"        
        self.dict_char_seq = {v:(i+1) for i,v in enumerate(seq_voc)}
        
        self.max_seq_len = 1200
        self.max_mol_len = 150 

    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        [atom.GetIsAromatic()])

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
        
    def mol_to_graph(self, mol):              
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom).reshape(1, -1)
            features.append( feature / (sum(feature)+1e-8 ))
        features = torch.FloatTensor(np.concatenate(features))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        edge_index = torch.LongTensor(edge_index).t()
        
        graph = Data(x=features, edge_index=edge_index)

        return graph
    


class _Dataset(Dataset,GraphDTAData):
    def __init__(self, df,pro_dir):
   
        self.df = df
        self.ids = self.df.index.values
        self.df_seq = pd.read_excel(pro_dir)
        self.gene2seq = self.df_seq.set_index('gene')['seq'].to_dict()
        self.register_init_feature()

    def __len__(self):
        return len(self.df)
        # load data
    
    def __getitem__(self, idx):
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        smiles = row['smiles']
        gene = row['gene']
        seq = self.gene2seq[gene]
        label = row['max']
        compound = self.get_compound(smiles)
        protein = self.get_protein(seq)
        return gene, compound, protein, label

#自定义colloate函数
def _collate(datalist):

    def padding(data):
        length = [len(i) for i in data]
        encoding = torch.zeros(len(data), 1200).long()
        for i in range(len(data)):
            encoding[i,:length[i]] = data[i]
        return encoding

    datalist = list(zip(*datalist))
    gene, compound, protein, affinity = datalist
    compound_batch = Batch.from_data_list(compound)
    protein_batch = padding(protein)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return gene,compound_batch, protein_batch, affinity_batch

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
