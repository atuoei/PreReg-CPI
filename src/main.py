# cli/train_dat3.py
import pandas as pd, os, torch
import argparse
from datasets import *
from training import Trainer

from importlib import import_module
from wrappers import build_wrapper
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g., DeepCPI, DAT3, GraphDTA")
    p.add_argument("--split", choices=["ck","compound","cluster"], required=True)
    p.add_argument("--mode", choices=["pretrain-finetune","from-scratch"], required=True)
    p.add_argument('--root_dir',default='../data')
    p.add_argument("--data_dir", default="regdata_cv")
    return p.parse_args()

def load_fold_tables(root_dir,data_dir, fold):
    tr = pd.read_csv(f"{root_dir}/{data_dir}/train.tsv", sep="\t")
    va = pd.read_csv(f"{root_dir}/{data_dir}/valid.tsv", sep="\t")
    te = pd.read_csv(f"{root_dir}/{data_dir}/test.tsv",  sep="\t")
    return (tr[tr.fold==fold].reset_index(drop=True),
            va[va.fold==fold].reset_index(drop=True),
            te[te.fold==fold].reset_index(drop=True))
def load_cla_tables(root_dir):
    tr = pd.read_csv(f'{root_dir}/cladata/clatrain.tsv',sep='\t')
    te = pd.read_csv(f'{root_dir}/cladata/clatest.tsv',sep='\t')
    return (tr,te)

def pretrain_then_finetune(model_name, split, data_dir,root_dir):
    
    df_clatr,df_clate = load_cla_tables(root_dir)
    # --- 1) classification pretrain ---
    adapter = import_module(f"datasets.{model_name}").Modeldataset(df_clatr, df_clate, df_clate, batch_size=128)
    train_loader, val_loader, test_loader = adapter.dataloaders()
    out_dir = f"checkpoints/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_cls = f"{out_dir}/cls_{split}.pt"

    cls_model = import_module(f'models.{model_name}').Model()                # BCE/AUC
    Trainer(dict(epochs=100, lr=1e-3, patience=8),mode = 'classification').fit(
        cls_model, train_loader, val_loader, ckpt_cls
    )
    for fold in range(5):
        df_tr, df_va, df_te = load_fold_tables(root_dir,data_dir, fold)

       

        # --- 2) regression finetune ---
        adapter = import_module(f"datasets.{model_name}").Modeldataset(df_tr, df_va, df_te, batch_size=128)
        train_loader, val_loader, test_loader = adapter.dataloaders()
        reg_wrapper = build_wrapper(model_name, pretrained=ckpt_cls)  # 载入分类权重
        Trainer(dict(epochs=100, lr=1e-4, patience=15),mode = 'regression').fit(
            reg_wrapper, train_loader, val_loader, f"{out_dir}/regft_{split}_fold{fold}.pt"
        )
        # test
        reg_wrapper.load_state_dict(torch.load(f"{out_dir}/regft_{split}_fold{fold}.pt", map_location="cpu"))
        genes, y, yhat, m = Trainer().test(reg_wrapper, test_loader)
        pd.DataFrame({"gene":genes,"label":y,"pred":yhat}).to_csv(f"{out_dir}/pred_regft_{split}_fold{fold}.csv", index=False)
        print("[reg-ft] test:", m)

def regress_from_scratch(model_name, split, data_dir,root_dir):
    for fold in range(5):
        df_tr, df_va, df_te = load_fold_tables(root_dir,data_dir, fold)
        adapter = import_module(f"datasets.{model_name}").Modeldataset(df_tr, df_va, df_te, batch_size=128)
        train_loader, val_loader, test_loader = adapter.dataloaders()
        # from scratch
        out_dir = f"checkpoints/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        model = import_module(f"models.{model_name}").Model()       
        save = f"{out_dir}/regs_{split}_fold{fold}.pt"
        Trainer(dict(epochs=100, lr=1e-3, patience=15)).fit(model, train_loader, val_loader, save)
        # test
        model.load_state_dict(torch.load(save, map_location="cpu"))
        genes, y, yhat, m = Trainer().test(model, test_loader)
        pd.DataFrame({"gene":genes,"label":y,"pred":yhat}).to_csv(f"{out_dir}/pred_regs_{split}_fold{fold}.csv", index=False)
        print("[reg-scratch] test:", m)
def main():
    args = parse_args()
    set_seed(42)
    if args.mode == "pretrain-finetune":
        pretrain_then_finetune(args.model, args.split, args.data_dir,args.root_dir)
    else:
        regress_from_scratch(args.model, args.split, args.data_dir,args.root_dir)

if __name__ == "__main__":
    main()