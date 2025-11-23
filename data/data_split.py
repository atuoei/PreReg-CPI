#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproductive-toxicity dataset preparation pipeline

Overview
1) Classification data: for each gene CSV in REP_class, perform a stratified split on 'class',
   then merge per-gene splits into cladata/{clatrain.tsv, clatest.tsv}.
2) Regression CV: for each gene CSV in REP_IC50, stratify by quintiles of 'max',
   hold out 20% test, then 5-fold CV on the remaining -> regdata_cv/{train/valid/test}.tsv
3) Chemical split: build a 5-fold split on unique SMILES (fold=0 from CV + test),
   with a 75/25 train/val split inside the train_val set -> regdata_chem/{train/valid/test}.tsv
4) Cluster split: compute Morgan fingerprints + KMeans clustering, split by clusters into 5 folds
   (train/val=3:1 within train_val) -> regdata_cluster/{train/valid/test}.tsv
"""

from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


# -------------------------- Global config -------------------------- #
SEED = 42
RADIUS = 2
N_BITS = 1024
N_CLUSTERS = 100
N_SPLITS = 5
VAL_RATIO_IN_TRAINVAL = 0.25  # for chemical/cluster splits inside train_val

# I/O layout (customize as needed)
DIR_CLASS = Path("REP_class")
DIR_REG = Path("REP_IC50")
OUT_CLASS = Path("cladata")
OUT_REG_CV = Path("regdata_cv")
OUT_REG_CHEM = Path("regdata_chem")
OUT_REG_CLUSTER = Path("regdata_cluster")


# -------------------------- Utilities -------------------------- #
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def smiles_to_fp(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=True
    )


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.warning(f"Failed to read CSV, skipping: {path} -> {e}")
        return None


# -------------------------- 1) Classification splits -------------------------- #
def build_classification_splits(src_dir: Path, out_dir: Path) -> None:
    """Per-gene stratified split (stratify='class'), then merge into global train/test."""
    ensure_dir(out_dir)
    tests, trains = [], []

    for file in src_dir.glob("*.csv"):
        gene = file.stem
        df = safe_read_csv(file)
        if df is None:
            continue

        if "class" not in df.columns:
            logging.warning(f"[Classification] Missing 'class' column, skip: {file}")
            continue

        try:
            df_train, df_test = train_test_split(
                df.reset_index(drop=True),
                test_size=0.2,
                stratify=df["class"],
                random_state=SEED,
            )
        except ValueError as e:
            logging.warning(
                f"[Classification] Stratified split failed ({gene}): {e} -> fallback to non-stratified split"
            )
            df_train, df_test = train_test_split(
                df.reset_index(drop=True),
                test_size=0.2,
                random_state=SEED,
            )

        trains.append(df_train.reset_index(drop=True))
        tests.append(df_test.reset_index(drop=True))

    if not trains or not tests:
        logging.error("[Classification] No splits generated. Check source directory/columns.")
        return

    df_train_all = pd.concat(trains, ignore_index=True)
    df_test_all = pd.concat(tests, ignore_index=True)

    df_train_all.to_csv(out_dir / "clatrain.tsv", sep="\t", index=False)
    df_test_all.to_csv(out_dir / "clatest.tsv", sep="\t", index=False)
    logging.info(f"[Classification] Saved: {out_dir/'clatrain.tsv'}, {out_dir/'clatest.tsv'}")


# -------------------------- 2) Regression CV (per-gene stratification) -------------------------- #
def stratify_into_quintiles(df: pd.DataFrame, value_col: str = "max") -> pd.DataFrame:
    """Create 5 quantile groups by `value_col`; fall back to equal sized groups if qcut fails."""
    df = df.copy()
    try:
        df["group"] = pd.qcut(df[value_col], q=5, labels=[f"Group {i+1}" for i in range(5)])
    except Exception:
        df = df.sort_values(by=value_col, ascending=True).reset_index(drop=True)
        n = len(df)
        base = n // 5
        rem = n % 5
        groups = []
        for i in range(5):
            groups += [f"Group {i+1}"] * base
        # deterministically assign remainders
        rng = np.random.default_rng(SEED)
        extra_idx = rng.choice(5, size=rem, replace=False)
        for i in extra_idx:
            groups.append(f"Group {i+1}")
        df["group"] = groups[:n]
    return df


def build_regression_cv(src_dir: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each gene CSV:
      - stratify by quintiles of 'max'
      - 20% external test split
      - 5-fold StratifiedKFold on the remaining set
    Merge all per-gene splits into global train/valid/test TSVs.
    """
    ensure_dir(out_dir)
    trains, vals, tests = [], [], []

    for file in src_dir.glob("*.csv"):
        gene = file.stem
        df = safe_read_csv(file)
        if df is None:
            continue
        if "max" not in df.columns:
            logging.warning(f"[Regression CV] Missing 'max' column, skip: {file}")
            continue

        df = stratify_into_quintiles(df, value_col="max")

        try:
            df_other, df_test = train_test_split(
                df, test_size=0.2, stratify=df["group"], random_state=SEED
            )
        except ValueError as e:
            logging.warning(
                f"[Regression CV] Stratified holdout failed ({gene}): {e} -> fallback to non-stratified holdout"
            )
            df_other, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

        df_other = df_other.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        tests.append(df_test)

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (tr_idx, va_idx) in enumerate(cv.split(df_other, df_other["group"])):
            df_tr, df_va = df_other.iloc[tr_idx].copy(), df_other.iloc[va_idx].copy()
            df_tr["fold"] = fold
            df_va["fold"] = fold
            trains.append(df_tr.reset_index(drop=True))
            vals.append(df_va.reset_index(drop=True))

    if not trains or not vals or not tests:
        logging.error("[Regression CV] No splits generated. Check source directory/columns.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_train_all = pd.concat(trains, ignore_index=True)
    df_valid_all = pd.concat(vals, ignore_index=True)
    df_test_all = pd.concat(tests, ignore_index=True)

    df_train_all.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    df_valid_all.to_csv(out_dir / "valid.tsv", sep="\t", index=False)
    df_test_all.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    logging.info(f"[Regression CV] Saved: {out_dir/'train.tsv'}, {out_dir/'valid.tsv'}, {out_dir/'test.tsv'}")

    return df_train_all, df_valid_all, df_test_all


# -------------------------- 3) Chemical split (unique SMILES) -------------------------- #
def build_chemical_split(
    df_train_cv: pd.DataFrame,
    df_valid_cv: pd.DataFrame,
    df_test_cv: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Use one CV fold example (fold=0) + test to assemble a deduplicated set by SMILES.
    Then perform 5-fold split over unique SMILES; within the train_val part of each fold,
    create a 75/25 train/val split.
    """
    ensure_dir(out_dir)

    df_train0 = df_train_cv[df_train_cv.get("fold", -1) == 0]
    df_valid0 = df_valid_cv[df_valid_cv.get("fold", -1) == 0]

    df_all = pd.concat([df_train0, df_valid0, df_test_cv], ignore_index=True)
    if "smiles" not in df_all.columns:
        logging.error("[Chemical split] Missing 'smiles' column. Aborting.")
        return

    df_all = df_all.drop_duplicates(subset="smiles").reset_index(drop=True)
    unique_smiles = df_all["smiles"].unique()

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    trains, vals, tests = [], [], []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(unique_smiles)):
        test_smiles = set(unique_smiles[test_idx])
        train_val_smiles = list(unique_smiles[train_val_idx])

        rng = np.random.default_rng(SEED + fold)
        rng.shuffle(train_val_smiles)
        n_val = int(len(train_val_smiles) * VAL_RATIO_IN_TRAINVAL)
        val_smiles = set(train_val_smiles[:n_val])
        train_smiles = set(train_val_smiles[n_val:])

        df_tr = df_all[df_all["smiles"].isin(train_smiles)].copy()
        df_va = df_all[df_all["smiles"].isin(val_smiles)].copy()
        df_te = df_all[df_all["smiles"].isin(test_smiles)].copy()

        for d in (df_tr, df_va, df_te):
            d["fold"] = fold

        trains.append(df_tr)
        vals.append(df_va)
        tests.append(df_te)

    df_train_all = pd.concat(trains, ignore_index=True)
    df_valid_all = pd.concat(vals, ignore_index=True)
    df_test_all = pd.concat(tests, ignore_index=True)

    df_train_all.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    df_valid_all.to_csv(out_dir / "valid.tsv", sep="\t", index=False)
    df_test_all.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    logging.info(f"[Chemical split] Saved: {out_dir/'train.tsv'}, {out_dir/'valid.tsv'}, {out_dir/'test.tsv'}")


# -------------------------- 4) Cluster split (fingerprints + KMeans) -------------------------- #
def build_cluster_split(df_source: pd.DataFrame, out_dir: Path) -> None:
    """
    Given a source DataFrame (recommended: union of chemical split train/valid/test),
    compute Morgan fingerprints -> KMeans clustering -> 5-fold split by clusters
    (train/val=3:1 within train_val).
    """
    ensure_dir(out_dir)

    if "smiles" not in df_source.columns:
        logging.error("[Cluster split] Missing 'smiles' column. Aborting.")
        return

    fps, keep_idx = [], []
    for i, smi in enumerate(df_source["smiles"]):
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(np.array(fp))
            keep_idx.append(i)

    if not fps:
        logging.error("[Cluster split] No valid SMILES to generate fingerprints.")
        return

    X = np.array(fps, dtype=np.uint8)
    df_valid = df_source.iloc[keep_idx].copy()

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    cluster_ids = kmeans.fit_predict(X)
    df_valid["cluster"] = cluster_ids

    clusters = df_valid["cluster"].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    trains, vals, tests = [], [], []
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(clusters)):
        test_clusters = set(clusters[test_idx])
        train_val_clusters = list(clusters[train_val_idx])

        rng = np.random.default_rng(SEED + 100 + fold)
        rng.shuffle(train_val_clusters)

        n_val = int(len(train_val_clusters) * VAL_RATIO_IN_TRAINVAL)
        val_clusters = set(train_val_clusters[:n_val])
        train_clusters = set(train_val_clusters[n_val:])

        df_tr = df_valid[df_valid["cluster"].isin(train_clusters)].copy()
        df_va = df_valid[df_valid["cluster"].isin(val_clusters)].copy()
        df_te = df_valid[df_valid["cluster"].isin(test_clusters)].copy()

        for d in (df_tr, df_va, df_te):
            d["fold"] = fold

        trains.append(df_tr)
        vals.append(df_va)
        tests.append(df_te)

    df_train_all = pd.concat(trains, ignore_index=True)
    df_valid_all = pd.concat(vals, ignore_index=True)
    df_test_all = pd.concat(tests, ignore_index=True)

    df_train_all.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    df_valid_all.to_csv(out_dir / "valid.tsv", sep="\t", index=False)
    df_test_all.to_csv(out_dir / "test.tsv", sep="\t", index=False)
    logging.info(f"[Cluster split] Saved: {out_dir/'train.tsv'}, {out_dir/'valid.tsv'}, {out_dir/'test.tsv'}")


# -------------------------- Main -------------------------- #
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(SEED)

    # 1) Classification
    if DIR_CLASS.exists():
        build_classification_splits(DIR_CLASS, OUT_CLASS)
    else:
        logging.warning(f"[Classification] Source directory not found: {DIR_CLASS} -> skipping")

    # 2) Regression CV
    if DIR_REG.exists():
        df_train_cv, df_valid_cv, df_test_cv = build_regression_cv(DIR_REG, OUT_REG_CV)
    else:
        logging.error(f"[Regression CV] Source directory not found: {DIR_REG} -> aborting")
        return

    if df_train_cv.empty or df_valid_cv.empty or df_test_cv.empty:
        logging.error("[Regression CV] Empty outputs -> stop subsequent steps")
        return

    # 3) Chemical split (based on CV fold=0 + test)
    build_chemical_split(df_train_cv, df_valid_cv, df_test_cv, OUT_REG_CHEM)

    # 4) Cluster split (prefer union of chemical split; fallback to CV union if needed)
    try:
        df_tr = pd.read_csv(OUT_REG_CHEM / "train.tsv", sep="\t")
        df_va = pd.read_csv(OUT_REG_CHEM / "valid.tsv", sep="\t")
        df_te = pd.read_csv(OUT_REG_CHEM / "test.tsv", sep="\t")
        df_union = (
            pd.concat([df_tr, df_va, df_te], ignore_index=True)
            .drop_duplicates(subset="smiles")
        )
        build_cluster_split(df_union, OUT_REG_CLUSTER)
    except Exception as e:
        logging.warning(
            f"[Cluster split] Could not use chemical split as source. "
            f"Fallback to CV union. Reason: {e}"
        )
        df_union = (
            pd.concat([df_train_cv, df_valid_cv, df_test_cv], ignore_index=True)
            .drop_duplicates(subset="smiles")
        )
        build_cluster_split(df_union, OUT_REG_CLUSTER)


if __name__ == "__main__":
    main()
