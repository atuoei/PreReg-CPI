import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




MODEL_PARAMS = {
    'drug_len': 1024,
    'prot_len': 1200,
    'filters': 64,
    'protein_dim': 20,
}

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(self.dim)[0]

class Model(nn.Module):
    """
    MONN for affinity prediction
    """
    def __init__(self, args=MODEL_PARAMS):
        super(Model, self).__init__()
        self.drug_len = int(args['drug_len'])
        self.prot_len = int(args['prot_len'])
        self.filters = int(args['filters'])
        self.protein_dim = int(args['protein_dim'])

        self.drug_encoder = nn.Sequential(
            nn.Linear(self.drug_len, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.protein_embedding = nn.Sequential(
            nn.Embedding(26, self.protein_dim),
            Transpose(1, 2),# convert to B,C,L
            nn.Dropout1d(0.2),
        )
        self.CNN = nn.ModuleList()
        for kernal_size in [10, 15, 20, 25]:
            self.CNN.append(nn.Sequential(
                nn.Conv1d(self.protein_dim, self.filters, kernal_size, padding=int(np.ceil((kernal_size-1)/2))),
                nn.BatchNorm1d(self.filters),
                nn.ReLU(),
                Max(-1),
            ))

        self.prot_encoder = nn.Sequential(
            nn.Linear(self.filters*4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(512+64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.apply(weights_init)

    def forward(self, input):
        drug_feature, protein_feature = input

        drug = self.drug_encoder(drug_feature)
        prot = self.protein_embedding(protein_feature)
        prot = torch.cat([encoder(prot) for encoder in self.CNN], dim=1)
        prot = self.prot_encoder(prot)

        feature = torch.cat([drug, prot], dim=1)
        affinity = self.fc_decoder(feature)
   

        return  affinity



