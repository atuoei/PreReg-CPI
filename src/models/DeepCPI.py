import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import torch
import numpy as np
import os

MODEL_PARAMS = {
    'mol_input_dim': 1024,
    'seq_input_dim': 640,
    'encode1': 1024,
    'encode2': 256,
    'output1': 512,
    'output2': 128,                 
    'output3': 32,
    }
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, args=MODEL_PARAMS):
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