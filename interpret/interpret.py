# -*- coding: utf-8 -*-
"""
DeepCPI-MLP (Ligand Fingerprint 1024 + Protein ESM 640) for Generating “Structural Alerts (SMARTS)”
Functions:
1 Uses Integrated Gradients (IG) to compute bit-level importance (for ligand bits only).
2 Applies Top-K bit masking (set to 0) to measure Δy robustness.
3 Maps important bits to RDKit SMARTS patterns (bit → (atom_idx, r) → subgraph), with optional neighborhood expansion up to r = 3 for visualization.

Dependencies:
torch, captum
rdkit
numpy, pandas
scipy, statsmodels
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor
from captum.attr import IntegratedGradients

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# ----------------------------
# model
# ----------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mol_input_dim = int(args["mol_input_dim"])
        self.seq_input_dim = int(args["seq_input_dim"])
        self.encode1 = int(args["encode1"])
        self.encode2 = int(args["encode2"])
        self.output1 = int(args["output1"])
        self.output2 = int(args["output2"])
        self.output3 = int(args['output3'])

        # ==== Compound Encoder ====
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

        # ==== Protein Encoder ====
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

        # ==== FiLM ====
        self.film_gamma = nn.Sequential(
            nn.Linear(self.encode2, self.encode2),
            nn.ReLU(),
            nn.Linear(self.encode2, self.encode2),
        )
        self.film_beta = nn.Sequential(
            nn.Linear(self.encode2, self.encode2),
            nn.ReLU(),
            nn.Linear(self.encode2, self.encode2),
        )

        # ==== Output Head ====
        self.Output = nn.Sequential(
            nn.Linear(self.encode2 * 2, self.output1),
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

    def forward(self, comp, prot):
        ca = self.CompoundEncoding(comp)   # [B, encode2]
        pa = self.ProteinEncoding(prot)    # [B, encode2]

        # --- FiLM: protein-conditioned modulation ---
        gamma = self.film_gamma(pa)        # [B, encode2]
        beta  = self.film_beta(pa)         # [B, encode2]
        ca_film = gamma * ca + beta        # [B, encode2]

        affinity = self.Output(torch.cat((ca_film, pa), dim=-1))  # [B, 1]
        return affinity


MODEL_PARAMS = {
    'mol_input_dim': 1024,
    'seq_input_dim': 640,
    'encode1': 1024,
    'encode2': 256,
    'output1': 512,
    'output2': 128,
    'output3': 32,
}


def load_model(checkpoint_path: Optional[str] = None) -> nn.Module:
    model = Model(MODEL_PARAMS)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get('state_dict', state)
        
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ----------------------------
# RDKit mapping
# ----------------------------
def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.SanitizeMol(mol)
    return mol


def morgan_bits_with_info(mol: Chem.Mol, radius: int = 2, nBits: int = 1024):
    """Return the bit vector (NumPy 0/1 float) and the corresponding bitInfo mapping."""
    bitInfo: Dict[int, List[Tuple[int, int]]] = {}
    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=nBits, useChirality=True, bitInfo=bitInfo
    )
    arr = np.zeros((nBits,), dtype=np.float32)  
    onbits = list(bv.GetOnBits())
    arr[onbits] = 1.0
    return arr, bitInfo


def env_to_smarts(mol: Chem.Mol, atom_idx: int, r: int,
                  expand_r_for_display: Optional[int] = None) -> str:
    """Convert the (central atom, radius r) environment to a SMARTS string; optionally expand the neighborhood by one additional radius for visualization."""
    use_r = max(r, expand_r_for_display) if expand_r_for_display is not None else r
    bond_env = list(Chem.FindAtomEnvironmentOfRadiusN(mol, use_r, atom_idx))  #  list
    atom_set = {int(atom_idx)}
    for bidx in bond_env:
        b = mol.GetBondWithIdx(int(bidx))
        atom_set.add(b.GetBeginAtomIdx())
        atom_set.add(b.GetEndAtomIdx())
    smarts = Chem.MolFragmentToSmarts(
        mol,
        atomsToUse=sorted(atom_set),
        bondsToUse=bond_env,
        isomericSmarts=True,   
    )
    return smarts


def bits_to_smarts_list(mol: Chem.Mol,
                        bitInfo: Dict[int, List[Tuple[int, int]]],
                        bits: Iterable[int],
                        expand_r_for_display: Optional[int] = 3) -> List[str]:
    """bit to SMART"""
    smarts_list: List[str] = []
    seen = set()
    for bit in bits:
        envs = bitInfo.get(int(bit), [])
        for (aidx, r) in envs:
            s = env_to_smarts(mol, aidx, r,
                              expand_r_for_display=expand_r_for_display)
            if s and s not in seen:
                seen.add(s)
                smarts_list.append(s)
    return smarts_list


# ----------------------------
# IG + sensitivity
# ----------------------------
@torch.no_grad()
def predict_scores(model: nn.Module, lig_bits: Tensor, prot_feat: Tensor) -> Tensor:
    out = model(lig_bits, prot_feat)
    if out.dim() > 1:
        out = out.squeeze(-1)
    return out


def compute_ig(model: nn.Module,
               lig_bits: Tensor,        # [N,1024], float32
               prot_feat: Tensor,       # [N,640], float32
               ig_steps: int = 64,
               target: Optional[int] = None,
               device: Optional[torch.device] = None) -> np.ndarray:
    """Convert a set of bits into a list of SMARTS strings (deduplicated)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    lig_bits = lig_bits.to(device).requires_grad_(True)
    prot_feat = prot_feat.to(device)  # 不求导

    def forward_only_lig(x_lig: Tensor, x_prot: Tensor) -> Tensor:
        return model(x_lig, x_prot)

    ig = IntegratedGradients(forward_only_lig)
    baseline = torch.zeros_like(lig_bits)

    attributions = ig.attribute(
        lig_bits,
        baselines=baseline,
        additional_forward_args=(prot_feat,),
        target=target,
        n_steps=ig_steps,
        internal_batch_size=None
    )
    return attributions.detach().cpu().numpy()


def occlusion_delta(model: nn.Module,
                    lig_bits: Tensor,        # [N,1024]
                    prot_feat: Tensor,       # [N,640]
                    topk_indices: List[np.ndarray],  
                    device: Optional[torch.device] = None) -> np.ndarray:
    """Mask the Top-K bits (set to 0) and compute Δy = y_orig - y_masked."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    lig_bits = lig_bits.to(device)
    prot_feat = prot_feat.to(device)

    with torch.no_grad():
        y_orig = predict_scores(model, lig_bits, prot_feat)  # [N]

    deltas = []
    for i in range(lig_bits.size(0)):
        x = lig_bits[i].clone()
        idx = topk_indices[i]
        if idx.size == 0:
            deltas.append(0.0)
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=x.device)
        x[idx_t] = 0.0  
        with torch.no_grad():
            y_mask = predict_scores(model, x.unsqueeze(0), prot_feat[i].unsqueeze(0)).item()
        deltas.append(float(y_orig[i].item() - y_mask))
    return np.asarray(deltas, dtype=np.float32)


# ----------------------------
# enrichment test
# ----------------------------
def summarize_alerts_per_protein(smiles_list: List[str],
                                 proteins: List[str],
                                 ig_attr: np.ndarray,     # [N,1024]
                                 deltas: np.ndarray,      # [N]
                                 topk_bits: List[np.ndarray],
                                 expand_r_for_display: int = 4,
                                 radius: int = 2,
                                 nBits: int = 1024) -> pd.DataFrame:

    rows = []
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        _, bitInfo = morgan_bits_with_info(mol, radius=radius, nBits=nBits)
        bits = topk_bits[i]
        smarts_list = bits_to_smarts_list(mol, bitInfo, bits,
                                          expand_r_for_display=expand_r_for_display)
        if not smarts_list:
            continue

        ig_i = ig_attr[i]
        pos_vals = ig_i[bits] if len(bits) > 0 else np.array([])
        pos_vals = pos_vals[pos_vals > 0]  
        mean_pos_ig = float(np.mean(pos_vals)) if pos_vals.size > 0 else 0.0

        for s in smarts_list:
            rows.append({
                "protein": proteins[i],
                "smarts": s,
                "example_smiles": smi,
                "mean_pos_ig_per_mol": mean_pos_ig,
                "delta_per_mol": float(deltas[i])
            })

    if not rows:
        return pd.DataFrame(columns=[
            "protein", "smarts", "n_hits", "mean_pos_ig", "mean_delta", "example_smiles"
        ])

    df = pd.DataFrame(rows)
    g = df.groupby(["protein", "smarts"], as_index=False).agg(
        n_hits=("smarts", "count"),
        mean_pos_ig=("mean_pos_ig_per_mol", "mean"),
        mean_delta=("delta_per_mol", "mean"),
        example_smiles=("example_smiles", "first")
    ).sort_values(
        ["protein", "n_hits", "mean_pos_ig", "mean_delta"],
        ascending=[True, False, False, False]
    )
    return g



def build_presence_matrix(smiles_list: List[str], smarts_list: List[str]) -> np.ndarray:
   
    mols = [smiles_to_mol(s) for s in smiles_list]
    patts = [Chem.MolFromSmarts(s) for s in smarts_list]
    M, N = len(smarts_list), len(smiles_list)
    X = np.zeros((N, M), dtype=np.int8)
    for j in range(M):
        if patts[j] is None:
            continue
        for i in range(N):
            m = mols[i]
            if m is None:
                continue
            if m.HasSubstructMatch(patts[j]):
                X[i, j] = 1
    return X


def enrichment_by_protein(smiles_list: List[str],
                          proteins: List[str],
                          y_pred: np.ndarray,            # [N]
                          y_true: Optional[np.ndarray],  # [N] or None
                          candidate_smarts: List[str],
                          top_q: float = 0.2,
                          fdr: float = 0.05,
                          use_true: bool = True,
                          higher_is_better: bool = True) -> pd.DataFrame:
 
    proteins = np.asarray(proteins)
    y = None
    if use_true and y_true is not None:
        y = np.asarray(y_true).reshape(-1)
    else:
        y = np.asarray(y_pred).reshape(-1)

    X = build_presence_matrix(smiles_list, candidate_smarts)  # [N,M]
    res_rows = []
    for p in np.unique(proteins):
        idx = np.where(proteins == p)[0]
        if idx.size < 10:
            continue
        y_p = y[idx]
        X_p = X[idx, :]

        thr = np.quantile(y_p, 1.0 - top_q) if higher_is_better else np.quantile(y_p, top_q)
        pos = (y_p >= thr).astype(np.int8) if higher_is_better else (y_p <= thr).astype(np.int8)
        neg = 1 - pos

        for j, s in enumerate(candidate_smarts):
            present = X_p[:, j]
            a = int(np.sum((present == 1) & (pos == 1)))
            b = int(np.sum((present == 1) & (neg == 1)))
            c = int(np.sum((present == 0) & (pos == 1)))
            d = int(np.sum((present == 0) & (neg == 1)))

            a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            OR, pval = fisher_exact([[a_, b_], [c_, d_]], alternative="greater")
            res_rows.append({
                "protein": p,
                "smarts": s,
                "a_pos_motif": a,
                "b_neg_motif": b,
                "c_pos_no_motif": c,
                "d_neg_no_motif": d,
                "odds_ratio": OR,
                "p_value": pval
            })

    df = pd.DataFrame(res_rows)
    if df.empty:
        return df
    _, qvals, _, _ = multipletests(df["p_value"].values, alpha=fdr, method="fdr_bh")
    df["q_value"] = qvals
    df = df.sort_values(["q_value", "odds_ratio"], ascending=[True, False])
    return df



# ----------------------------
# visiualize
# ----------------------------
def draw_smarts_on_smiles(smiles: str, smarts: str, out_png: str, size=(360, 280)):
    mol = smiles_to_mol(smiles)
    patt = Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return
    hit = mol.GetSubstructMatch(patt)
    highlight_atoms = list(hit) if hit else []
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=highlight_atoms)
    d2d.FinishDrawing()
    d2d.WriteDrawingText(out_png)


# ----------------------------
# main process
# ----------------------------
def _to_float_tensor(x):
    
    if isinstance(x, torch.Tensor):
        return x.detach().clone().float()
    return torch.as_tensor(x, dtype=torch.float32)


def explain_and_summarize(model: nn.Module,
                          X_lig: np.ndarray,        # [N,1024]
                          X_prot: np.ndarray,       # [N,640]
                          smiles_list: List[str],
                          protein_list: List[str],
                          y_true: Optional[np.ndarray],  # None
                          ig_steps: int = 64,
                          topk: int = 10,
                          expand_r_for_display: int = 3,
                          device: Optional[str] = None,
                          do_enrichment: bool = True,
                          top_q: float = 0.2,
                          fdr: float = 0.05,
                          out_dir: Optional[str] = None,
                          focus_summary_on_high: bool = True,     
                          y_true_higher_is_better: bool = True): 
    device_t = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xl = _to_float_tensor(X_lig).to(device_t)
    Xp = _to_float_tensor(X_prot).to(device_t)

    # 1) predict
    with torch.no_grad():
        y_pred = predict_scores(model.to(device_t).eval(), Xl, Xp).cpu().numpy()

    # 1.1 
    if y_true is not None:
        y_ref = np.asarray(y_true).reshape(-1)
        thr = np.quantile(y_ref, 1.0 - top_q) if y_true_higher_is_better else np.quantile(y_ref, top_q)
        high_mask = (y_ref >= thr) if y_true_higher_is_better else (y_ref <= thr)
    else:
        thr = np.quantile(y_pred, 1.0 - top_q)
        high_mask = (y_pred >= thr)

    idx_high = np.where(high_mask)[0]

    # 2) IG 
    ig_attr = compute_ig(model, Xl, Xp, ig_steps=ig_steps, target=None, device=device_t)  # [N,1024]

    # 3) Top-K
    on_mask = (Xl.detach().cpu().numpy() > 0.5)
    topk_indices: List[np.ndarray] = []
    for i in range(ig_attr.shape[0]):
        scores = ig_attr[i].copy()
        scores[~on_mask[i]] = -1e12
        scores[scores <= 0] = -1e12
        idx = np.argsort(-scores)[:topk]
        idx = idx[scores[idx] > -1e11]
        topk_indices.append(idx)

    #  Δy
    deltas = occlusion_delta(model, Xl, Xp, topk_indices, device=device_t)

    # 4) summary
    if focus_summary_on_high:
        ig_use = ig_attr[idx_high]
        deltas_use = deltas[idx_high]
        topk_use = [topk_indices[i] for i in idx_high]
        smiles_use = [smiles_list[i] for i in idx_high]
        proteins_use = [protein_list[i] for i in idx_high]
    else:
        ig_use, deltas_use, topk_use = ig_attr, deltas, topk_indices
        smiles_use, proteins_use = smiles_list, protein_list

    df_alerts = summarize_alerts_per_protein(
        smiles_list=smiles_use,
        proteins=proteins_use,
        ig_attr=ig_use,
        deltas=deltas_use,
        topk_bits=topk_use,
        expand_r_for_display=expand_r_for_display,
        radius=2,
        nBits=1024
    )
    # === put example_smiles and its true&predicted affinity into the file  ===
    df_meta_for_alerts = pd.DataFrame({
        "smiles": smiles_list,
        "protein": protein_list,
        "y_true": y_true if y_true is not None else np.full(len(smiles_list), np.nan),
        "y_pred": y_pred,
    })
    df_alerts = df_alerts.merge(
        df_meta_for_alerts,
        left_on=["example_smiles", "protein"],
        right_on=["smiles", "protein"],
        how="left"
    ).drop(columns=["smiles"]).rename(columns={
        "y_true": "example_y_true",
        "y_pred": "example_y_pred"
    })

    result = {
        "alerts": df_alerts, 
        "pred": pd.DataFrame({
            "smiles": smiles_list,
            "protein": protein_list,
            "y_pred": y_pred,
            "y_true": y_true if y_true is not None else np.full(len(smiles_list), np.nan)
        })
    }


    # 5) enrichment test
    if do_enrichment and not df_alerts.empty:
        candidate = df_alerts["smarts"].drop_duplicates().tolist()
        df_enr = enrichment_by_protein(
            smiles_list=smiles_list,
            proteins=protein_list,
            y_pred=y_pred,
            y_true=y_true,
            candidate_smarts=candidate,
            top_q=top_q,
            fdr=fdr,
            use_true=(y_true is not None),
            higher_is_better=y_true_higher_is_better
        )
        result["enrichment"] = df_enr

    # 6) output file
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        result["alerts"].to_csv(os.path.join(out_dir, "alerts_summary.csv"), index=False)
        result["pred"].to_csv(os.path.join(out_dir, "pred_scores.csv"), index=False)
        if "enrichment" in result:
            result["enrichment"].to_csv(os.path.join(out_dir, "enrichment.csv"), index=False)

                
        try:
            df_top = result["alerts"].groupby("protein", as_index=False).head(10).copy()

            # 
            def fmt(v):
                try:
                    if pd.isna(v): return "NA"
                    return f"{float(v):.2f}"
                except Exception:
                    return "NA"

            df_top["true_str"] = df_top["example_y_true"].apply(fmt)
            df_top["pred_str"] = df_top["example_y_pred"].apply(fmt)

            # 
            df_top["dup_idx"] = df_top.groupby(["protein", "true_str", "pred_str"]).cumcount()

            for _, row in df_top.iterrows():
                suffix = f"-{int(row['dup_idx'])}" if int(row["dup_idx"]) > 0 else ""
                fn = f"{row['protein']}_true{row['true_str']}_pred{row['pred_str']}{suffix}.png"
                out_path = os.path.join(out_dir, fn)
                draw_smarts_on_smiles(row["example_smiles"], row["smarts"], out_path)
        except Exception as e:
            print(f"[WARN]：{e}")



    return result


if __name__ == "__main__":
  
    model = load_model(checkpoint_path='regmodel_DTA/model.pth')

    # 2) Data import
    df_data = pd.read_csv('../data/regdata_full/train.tsv', sep='\t')
    df_profeat = pd.read_csv('../Features/esm_150m.csv', index_col=0)
    df_proteins = pd.read_csv('../Predict/predict_targets.csv')
    proteins = df_proteins['task'].tolist()
    for protein in proteins:
        df_smi = df_data[df_data['gene'] == protein]
        y_true = df_smi['max'].values
        smiles_list = df_smi['smiles'].tolist()
        N = len(smiles_list)

        # 3) Generate 1024-bit Morgan fingerprints (radius=2), consistent with preprocessing during training
        X_lig = []
        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            if mol is None:
                X_lig.append(np.zeros((1024,), dtype=np.float32))
            else:
                bits, _ = morgan_bits_with_info(mol, radius=2, nBits=1024)
                X_lig.append(bits.astype(np.float32))
        X_lig = np.vstack(X_lig)  # [N,1024] numpy array

        # 4) Protein features (keep as numpy array; converted to tensor inside function)
        rng = df_profeat.loc[protein, :].values.reshape(1, 640)   # (1, 640)
        X_prot = np.repeat(rng, N, axis=0)                        # (N, 640)

        # 5) Run explanation and summarization
        out = explain_and_summarize(
            model=model,
            X_lig=X_lig,
            X_prot=X_prot,
            smiles_list=smiles_list,
            protein_list=[protein] * N,
            y_true=y_true,
            ig_steps=64,
            topk=10,
            expand_r_for_display=3,
            device=None,          # "cuda" / "cpu"
            do_enrichment=True,
            top_q=0.33,
            fdr=0.05,
            out_dir=f"./single_target/{protein}_alerts"
        )

        # 6) Print brief results
        print("=== ALERTS (head) ===")
        print(out["alerts"].head())
        if "enrichment" in out:
            print("=== ENRICHMENT (head) ===")
            print(out["enrichment"].head())
        print("=== PRED (head) ===")
        print(out["pred"].head())
