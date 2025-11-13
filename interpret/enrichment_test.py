# pip install rdkit-pypi pandas numpy scipy statsmodels
from rdkit import Chem
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ===== 1)  SMARTS matrix =====
def smarts_hit_matrix(df: pd.DataFrame, alerts: list, smiles_col='smiles'):
    patt = [Chem.MolFromSmarts(s) for s in alerts]
    names = [f"A{idx+1}" for idx in range(len(alerts))]
    hits = []
    for smi in df[smiles_col].astype(str).values:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            hits.append([0]*len(patt))
            continue
        hits.append([int(m.HasSubstructMatch(p)) for p in patt])
    H = pd.DataFrame(hits, columns=names, index=df.index)
    H.attrs["alert_map"] = dict(zip(names, alerts))   
    return H

# ===== 2) Positive vs Negative enrichment test =====
def enrichment_pos_vs_neg(df: pd.DataFrame, H: pd.DataFrame,
                          label_col='Label', pos_tag='Positive', neg_tag='Negative',
                          min_group_n: int = 3):
    # only keep two types
    sub = df[df[label_col].isin([pos_tag, neg_tag])].copy()
    lab = sub[label_col].values
    if len(sub) == 0:
        raise ValueError("fail to find the Positive/Negative sample")

    results = []
    for col in H.columns:
        hit = H.loc[sub.index, col].astype(int).values

        a = int(((lab==pos_tag) & (hit==1)).sum())   # Positive hit
        b = int(((lab==pos_tag) & (hit==0)).sum())   # Positive non hit
        c = int(((lab==neg_tag) & (hit==1)).sum())   # Negative hit
        d = int(((lab==neg_tag) & (hit==0)).sum())   # Negative non hit

        high_n = a + b
        low_n  = c + d
        if high_n < min_group_n or low_n < min_group_n:
            
            continue

        # Haldane–Anscombe +0.5
        OR = ((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
        _, p = fisher_exact([[a,b],[c,d]], alternative='greater')  

        results.append({
            "alert": col, "OR": OR, "p": p,
            "a_pos_hit": a, "b_pos_nohit": b,
            "c_neg_hit": c, "d_neg_nohit": d,
            "pos_n": high_n, "neg_n": low_n
        })

    out = pd.DataFrame(results)
    if len(out):
        out["q"] = multipletests(out["p"], method="fdr_bh")[1]

        out["SMARTS"] = out["alert"].map(H.attrs["alert_map"])
    return out.sort_values(["q","OR"], ascending=[True, False])


df_smi = pd.read_excel('../Predict/tox_smi_dedup.xlsx')

df_alert = pd.read_csv('broad_target/broad_from_alerts.csv')

alerts = df_alert['smarts'].tolist()
H = smarts_hit_matrix(df_smi, alerts, smiles_col='new_smiles')
enrich = enrichment_pos_vs_neg(df_smi, H, label_col='Label', pos_tag='Positive', neg_tag='Negative')
enrich.to_csv('enrich_broad.csv',index=False)
sig = enrich.query("OR>1.5 and p<0.05")[["alert","SMARTS","OR","p","q","pos_n","neg_n","a_pos_hit","c_neg_hit"]]
sig.to_csv('sig_enrich_broad.csv',index=False)

