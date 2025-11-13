import pandas as pd
import torch
from torch._prims_common import RETURN_TYPE
from models import *
from importlib import import_module
def build_wrapper(model_name, pretrained=None):
    pretrained_dict = torch.load(pretrained,map_location='cpu')  
    if model_name == 'FusionDTA':
        # pretrain 
        model = import_module(f"models.{model_name}").Model()
        model_dict = model.state_dict()

        # filter（eg'out_fc2', 'out_fc3'）
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not (
            k.startswith('out_fc2') or k.startswith('out_fc3'))}

        # updata
        model_dict.update(filtered_dict)
       
        model = model.load_state_dict(model_dict)
        return model

    elif model_name == "AttentionDTA":
        model = import_module(f"models.{model_name}").Model()
         
        model_dict = model.state_dict()  

        
        pretrained_dict = {k: v for k, v in pretrained_dict.items()   
                        if k not in ['fc3.weight', 'fc3.bias', 'out.weight', 'out.bias']}  

         
        model_dict.update(pretrained_dict)  
        model.load_state_dict(model_dict)  
        return model
    elif model_name == 'DeepConvDTI':

        model = import_module(f"models.{model_name}").Model()
    

        if 'model_state_dict' in pretrained_dict:
            pretrained_state = pretrained_dict['model_state_dict']
        else:
            pretrained_state = pretrained_dict  

       
        filtered_state = {k: v for k, v in pretrained_state.items() if not k.startswith('fc_decoder')}

        model.load_state_dict(filtered_state, strict=False)
        return model
    
    elif model_name == 'GraphDTA':

        model = import_module(f"models.{model_name}").Model()

        model_dict = model.state_dict()

        filtered_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not (k.startswith("fc2.") or k.startswith("out."))
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)  
        return model
    elif model_name == 'DrugBAN':
        model = import_module(f"models.{model_name}").Model()
        model_dict = model.state_dict()  

        filtered_dict = {}
        for k, v in pretrained_dict.items():
           
            if k.startswith('mlp_classifier.fc3')  or \
            k.startswith('mlp_classifier.bn3')  or \
            k.startswith('mlp_classifier.fc4'):
                continue
            if k in model_dict:
                filtered_dict[k] = v

        # filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('mlp_classifier.')}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        return model
    elif model_name == 'DeepCPI':

        model = import_module(f"models.{model_name}").Model()
        model_dict = model.state_dict()  
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'Output' not in k} # 过滤掉 Output 模块中的最后两层
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
