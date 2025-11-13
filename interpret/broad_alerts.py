# -*- coding: utf-8 -*-
"""
Identify "broad-spectrum" SMARTS patterns based only on alerts_summary.csv (no statistical test).

Definitions (configurable):
- Within each target: signal = ReLU(mean_pos_ig) * ReLU(mean_delta)
- Strong signal criterion: within each target, signal percentile >= activation_pct (default 0.60)
- Broad-spectrum: covered_targets >= coverage_thr (default 3) 
                  AND mean(signal_pct) >= mean_sig_thr (default 0.55)
- Additional: Breadth (normalized entropy) >= breadth_thr (default 0.30)

Input:
Recursively search for **/alerts_summary.csv 
(required columns: protein, smarts, n_hits, mean_pos_ig, mean_delta, example_smiles)

Output:
- across_from_alerts/all_smarts_summary.csv           (cross-target summary table)
- across_from_alerts/broad_spectrum_from_alerts.csv   (list of broad-spectrum SMARTS)
- across_from_alerts/signal_matrix.csv                (Protein × SMARTS signal matrix; for heatmap)

Usage:
python broad_from_alerts.py --root . --outdir across_from_alerts \
    --activation_pct 0.60 --coverage_thr 3 --mean_sig_thr 0.55 --breadth_thr 0.30
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def load_alerts(root="."):
    """Recursively read all alerts_summary.csv files and standardize column names and types."""
    paths = glob.glob(os.path.join(root, "**", "alerts_summary.csv"), recursive=True)
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        need = ["protein","smarts","n_hits","mean_pos_ig","mean_delta"]
        if not set(need).issubset(df.columns):
            print(f"[WARN] Missing required columns, skipped {p}")
            continue
        # Keep only required columns; retain example_smiles if available
        keep = ["protein","smarts","n_hits","mean_pos_ig","mean_delta"]
        if "example_smiles" in df.columns:
            keep.append("example_smiles")
        df = df[keep].copy()
        # Convert data types
        for c in ["n_hits","mean_pos_ig","mean_delta"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["protein"] = df["protein"].astype(str)
        df["smarts"] = df["smarts"].astype(str).str.strip()
        df = df[(df["smarts"]!="") & df["smarts"].notna()]
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No alerts_summary.csv found under the specified root directory.")
    D = pd.concat(rows, ignore_index=True)
    # For duplicate (protein, smarts), keep the one with the highest mean_pos_ig * mean_delta
    D["signal_raw"] = relu(D["mean_pos_ig"]) * relu(D["mean_delta"])
    D = (D.sort_values(["protein","smarts","signal_raw"], ascending=[True,True,False])
           .drop_duplicates(["protein","smarts"], keep="first"))
    return D

def add_within_protein_percentile(D, col="signal_raw", out_col="signal_pct"):
    """Within each protein, compute percentile (0–1) for the given column; fill missing with 0."""
    D = D.copy()
    D[out_col] = 0.0
    for p, g in D.groupby("protein"):
        x = g[col].fillna(0).values
        if len(x)==0:
            D.loc[g.index, out_col] = 0.0
            continue
        # 百分位：x 的秩 / (n-1)，常规处理重复值用平均秩
        ranks = pd.Series(x).rank(method="average").values
        denom = max(len(x)-1, 1)
        pct = (ranks - 1) / denom
        D.loc[g.index, out_col] = pct
    return D

def summarize_across_targets(D, activation_pct=0.60):
    """
    Summarize SMARTS across targets:
    Define "active" SMARTS in a target as those with signal_pct >= activation_pct.
    Return SMARTS-level summary table and (protein × SMARTS) signal matrix data.
    """
    D = D.copy()
    D["active"] = (D["signal_pct"] >= float(activation_pct)).astype(int)

    rows = []
    for s, g in D.groupby("smarts"):
        proteins_all = sorted(g["protein"].unique().tolist())
        g_act = g[g["active"]==1]
        proteins_act = sorted(g_act["protein"].unique().tolist())
        coverage = len(proteins_act)
        mean_sig = float(g_act["signal_pct"].mean()) if coverage>0 else 0.0
        med_sig  = float(g_act["signal_pct"].median()) if coverage>0 else 0.0
        mean_hits = float(g_act["n_hits"].mean()) if coverage>0 else 0.0
        mean_ig   = float(g_act["mean_pos_ig"].mean()) if coverage>0 else 0.0
        mean_dy   = float(g_act["mean_delta"].mean()) if coverage>0 else 0.0

        # Breadth: normalized entropy of signal_pct distribution among active proteins
        breadth = 0.0
        if coverage > 1:
            w = g_act["signal_pct"].fillna(0).to_numpy(dtype=float)
            w = np.maximum(w, 0)
            if w.sum() > 0:
                p = w / w.sum()
                p = p[p>0]
                H = float(-(p*np.log(p+1e-12)).sum())
                breadth = float(H / np.log(len(p)))  # 0~1
        rows.append({
            "smarts": s,
            "coverage": coverage,
            "proteins_active": "|".join(proteins_act),
            "proteins_all_seen": "|".join(proteins_all),
            "mean_signal_pct": mean_sig,
            "median_signal_pct": med_sig,
            "breadth_score": breadth,
            "mean_n_hits": mean_hits,
            "mean_pos_ig_active": mean_ig,
            "mean_delta_active": mean_dy
        })
    M = pd.DataFrame(rows).sort_values(
        ["coverage","mean_signal_pct","breadth_score"], ascending=[False,False,False]
    )
    return M

def build_signal_matrix(D, selected_smarts):
    """
    Construct a Protein × SMARTS matrix (values = signal_pct).
    For duplicate protein-SMARTS pairs, keep the maximum value.
    """
    P = sorted(D["protein"].unique().tolist())
    idx_p = {p:i for i,p in enumerate(P)}
    Mlist = selected_smarts
    idx_m = {m:j for j,m in enumerate(Mlist)}
    mat = np.zeros((len(P), len(Mlist)), dtype=float)
    for _, r in D.iterrows():
        p, s = r["protein"], r["smarts"]
        if p in idx_p and s in idx_m:
            i, j = idx_p[p], idx_m[s]
            mat[i,j] = max(mat[i,j], float(r["signal_pct"]))
    return P, Mlist, mat


import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(P, Mlist, mat, out_png, cmap="magma", zero_as_white=True, reverse=False):
    """
    Plot a heatmap (SMARTS on the y-axis, Proteins on the x-axis)
    - cmap: color map name
    - zero_as_white: if True, cells with <=0 values are shown as white
    - reverse: if True, use reversed color map
    """

    vmax = np.percentile(mat[mat > 0], 95) if np.any(mat > 0) else None
    data = np.ma.masked_where(mat <= 0, mat) if zero_as_white else mat
    cmap_name = f"{cmap}_r" if reverse else cmap
    cmap_obj = plt.get_cmap(cmap_name)
    if zero_as_white:
        try:
            cmap_obj = cmap_obj.with_extremes(bad="#FFFFFF")
        except Exception:
            try:
                cmap_obj.set_bad("#FFFFFF")
            except Exception:
                pass

    plt.figure(figsize=(max(8, 0.20*len(P)+3), max(6, 0.18*len(Mlist)+3)))
    im = plt.imshow(
        data.T, aspect="auto", interpolation="nearest",
        cmap=cmap_obj, vmin=None if zero_as_white else 0, vmax=vmax
    )

    # x axis = Protein
    plt.xticks(ticks=np.arange(len(P)), labels=[str(x) for x in P], rotation=90, fontsize=8)
    # y axis = SMARTS
    plt.yticks(ticks=np.arange(len(Mlist)), labels=[str(x) for x in Mlist], fontsize=8)

    plt.xlabel("Protein")
    plt.ylabel("SMARTS (selected)")
    plt.title("Signal percentile heatmap (per protein)")

    cb = plt.colorbar(im)
    cb.set_label("signal_pct (0~1)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./single_target", help="Root directory to recursively search for alerts_summary.csv")
    ap.add_argument("--outdir", default="broad_targets")

    # Strong-signal definition (within each target)
    ap.add_argument("--activation_pct", type=float, default=0.60, help="Percentile threshold (0–1) for signal within each target")

    # Broad-spectrum filtering
    ap.add_argument("--coverage_thr", type=int, default=8, help="Minimum number of targets where the SMARTS is classified as active")
    ap.add_argument("--mean_sig_thr", type=float, default=0.6, help="Lower bound (0–1) for mean(signal_pct) among active targets")
    ap.add_argument("--breadth_thr", type=float, default=0.30, help="Lower bound (0–1) for normalized entropy, representing signal breadth")

    # Heatmap settings
    ap.add_argument("--heatmap_top", type=int, default=50, help="Number of motifs to display in the heatmap (sorted by coverage → mean signal)")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Read and calculate signals
    D = load_alerts(args.root)
    D = add_within_protein_percentile(D, col="signal_raw", out_col="signal_pct")

    # 2) Aggregate results (based only on alerts)
    M = summarize_across_targets(D, activation_pct=args.activation_pct)
    M.to_csv(os.path.join(args.outdir, "all_smarts_summary.csv"), index=False)

    # Identify broad-spectrum motifs
    broad = M[
        (M["coverage"] >= args.coverage_thr) &
        (M["breadth_score"] >= args.breadth_thr)
    ].sort_values(["coverage", "mean_signal_pct"], ascending=[False, False])

    # Save results
    broad.to_csv(os.path.join(args.outdir, "broad_from_alerts.csv"), index=False)

    # 4) Generate a Protein × SMARTS matrix (for heatmap or downstream use)
    top_smarts = broad["smarts"].head(args.heatmap_top).tolist()
    if len(top_smarts) == 0:
        # If no broad-spectrum SMARTS are found, fallback to top motifs by coverage
        top_smarts = (M.sort_values(["coverage", "mean_signal_pct"], ascending=[False, False])
                        ["smarts"].head(args.heatmap_top).tolist())
        print("[INFO] Current broad-spectrum thresholds are too strict; heatmap will show top motifs by coverage instead.")

    P, Mlist, mat = build_signal_matrix(D, top_smarts)

    # Save matrix
    pd.DataFrame(mat, index=P, columns=Mlist)\
      .to_csv(os.path.join(args.outdir, "signal_matrix.csv"))

    # Plot heatmap
    try:
        plot_heatmap(P, Mlist, mat, os.path.join(args.outdir, "heatmap_from_alerts.png"))
    except Exception as e:
        print(f"[WARN] {e}")

    print(f"[OK] summary: {len(M)} broad: {len(broad)}")
    print(f"[OK] files saved in {args.outdir}/")
    print(" - all_smarts_summary.csv")
    print(" - broad_from_alerts.csv")
    print(" - signal_matrix.csv")
    print(" - heatmap_from_alerts.png")

if __name__ == "__main__":
    main()