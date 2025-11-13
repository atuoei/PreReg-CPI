import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import pickle
import torch
import rdkit.Chem as Chem
from rdkit.Chem import AllChem,MACCSkeys
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import logging
from transformers import logging as hf_logging



def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.mol_input_dim = int(args["mol_input_dim"])
        self.seq_input_dim = int(args["seq_input_dim"])
        self.encode1 = int(args["encode1"])
        self.encode2 = int(args["encode2"])
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])
        
        """Compound Encoding Module"""
        self.CompoundEncoding = nn.Sequential(
            nn.Linear(self.mol_input_dim, self.encode1), 
            nn.BatchNorm1d(self.encode1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encode1, self.encode2),
            nn.BatchNorm1d(self.encode2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        """Protein Encoding Module"""
        self.ProteinEncoding = nn.Sequential(
            nn.Linear(self.seq_input_dim, self.encode1), 
            nn.BatchNorm1d(self.encode1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.encode1, self.encode2),
            nn.BatchNorm1d(self.encode2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        """Output Module"""
        self.Output = nn.Sequential(
            nn.Linear(self.encode2*2, self.output1),
            nn.BatchNorm1d(self.output1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output1, self.output2),
            nn.BatchNorm1d(self.output2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output2, self.output3),
            nn.BatchNorm1d(self.output3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output3, 1),
        )

        self.apply(weights_init)

    def forward(self, comp,prot):

        ca = self.CompoundEncoding(comp)
        pa = self.ProteinEncoding(prot)
        # protein = torch.cat((pa,bind_prompt),dim=-1)
        affinity = self.Output(torch.cat((ca, pa), dim=-1))


        return affinity


# 自定义collate函数
def batch_data_process_DeepCPI(datalist):
    if len(datalist) == 1:
        datalist = datalist * 2
        
    datalist = list(zip(*datalist))
    cas,target,compound, protein = datalist
    # # 确保compound是一个张量，进行squeeze操作


    compound_batch = torch.Tensor(compound)
    # compound_batch = compound_batch.squeeze(1)
    protein_batch = torch.Tensor(protein)
    return cas,target,compound_batch, protein_batch


#Dataset
from torch.utils.data import Dataset,DataLoader
class AffinityDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.ids = self.df.index.values
        self.df_prow2v = pd.read_csv('../Features/esm_150m.csv',index_col=0)
    def __len__(self):
        return len(self.df)
    
    def smiles_to_Morgan(self,smi, radius=2, n_bits=1024):
        mol = Chem.MolFromSmiles(smi)
        # finger = MACCSkeys.GenMACCSKeys(mol)
        finger = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits,useChirality=True))
        return finger
    
    def gene2feat(self,gene):
        feat = self.df_prow2v.loc[gene,:].values
        return feat

    def __getitem__(self, idx):
        idx = self.ids[idx]
        row = self.df.iloc[idx]
        cas = row['cas']
        smi = row['smiles']
        gene = row['gene']
        
        Morgan = self.smiles_to_Morgan(smi)
        prot_esm = self.gene2feat(gene)        
        
        return cas,gene,Morgan, prot_esm


MODEL_PARAMS = {
    'mol_input_dim': 1024,
    'seq_input_dim': 640,
    'encode1': 1024,
    'encode2': 256,
    'output1': 512,
    'output2': 128,                 
    'output3': 32,
}


# 直接预测
# cosmetic chem
import pandas as pd
df_full = pd.read_excel('tox_smi_dedup.xlsx')
df_fullp = df_full[df_full['Responsive'=='Positive']].reset_index(drop=True)
df_fulln = df_full[df_full['Responsive'=='Negative']].reset_index(drop=True)
df_reggene = pd.read_csv('predict_targets.csv')
reggenes = df_reggene['task'].tolist()
df_positive = []
for i,row in df_fullp.iterrows():
    cass = [row['CAS Number']]*len(reggenes)
    smiless = [row['smiles']]*len(reggenes)
    df = pd.DataFrame({'cas':cass,'smiles':smiless,'gene':reggenes})
    df_positive.append(df)
df_positive = pd.concat(df_positive)
df_positive.reset_index(drop=True,inplace=True)

df_negative = []
for i,row in df_fulln.iterrows():
    cass = [row['CAS Number']]*len(reggenes)
    smiless = [row['smiles']]*len(reggenes)
    df = pd.DataFrame({'cas':cass,'smiles':smiless,'gene':reggenes})
    df_negative.append(df)
df_negative = pd.concat(df_negative)
df_negative.reset_index(drop=True,inplace=True)
# df_postive df_negtive
for name,df in {'positive':df_positive,'negative':df_negative}:
    full_set = AffinityDataset(df)
    full_loader = DataLoader(full_set, 
                        batch_size=128, 
                        shuffle=False, 
                        collate_fn =batch_data_process_DeepCPI,
                        num_workers=0, 
                        # persistent_workers=False, 
                        drop_last=False, 
                        pin_memory=False)
    # loaddir = './CLA_pretrain/bindchembl640_clamodel'
    lpath = 'regmodel_DTA/model.pth'
    pred_dic={}
    r2s=[]
    rmses=[]
    for fold in range(1):


        #测试
        model = Model(MODEL_PARAMS)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(f'{lpath}'))
        model = model.to(device)
        model.eval()
        outputs = []
        cass = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(full_loader):
                # print(batch_idx)
                cas,target,compound, protein= batch
                # features = [item.to(device) for item in feature_tuple]
                compound = compound.to(device)
                protein = protein.to(device)
                output = model(compound,protein)
                
                output = output.cpu().numpy().reshape(-1)
    
                cass.extend(cas)
                outputs.extend(output)
                targets.extend(target)
        

        pred_dic['cas'] = cass
        pred_dic['task'] = [t for t in targets]

        pred_dic[fold] = [s.item() for s in outputs]
        del model
        torch.cuda.empty_cache()
    df_pred = pd.DataFrame(pred_dic)

    df_pivot = df_pred.pivot(index='cas', columns='task', values=0)
    # df_pivot.to_excel('food_mtl.xlsx')
    df_pivot.to_excel(f'{name}_regpred.xlsx')

    