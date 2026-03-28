#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import pairwise_distances

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from Net import Net  # your architecture

BATCH_SIZE = 256


# -------------------------
# Helpers
# -------------------------

def set_reproducible_inference(seed: int = 1234) -> None:
    """
    Make inference as deterministic as PyTorch allows.
    Call ONCE at the start of main() before loading models / running predict.
    """
    import os
    import random
    import numpy as np
    import torch
    from torch.backends import cudnn

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for deterministic cublas on CUDA

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # reduce numerical variability on GPU
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sanitize_column_name(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_smiles_column(df: pd.DataFrame) -> str:
    if "SMILES" in df.columns:
        return "SMILES"
    lower = {c.lower(): c for c in df.columns}
    for key in ["smiles", "canonical_smiles", "isomeric_smiles"]:
        if key in lower:
            return lower[key]
    raise ValueError("Could not infer SMILES column. Please name it 'SMILES' (recommended).")


def load_noheader_list_csv(path: Path) -> List[str]:
    """Reads a 1-column CSV list with NO header."""
    df = pd.read_csv(path, header=None)
    vals = df[0].astype(str).tolist()
    vals = [sanitize_column_name(v) for v in vals]
    return dedupe_preserve_order(vals)


def rdkit_available_descriptor_names() -> List[str]:
    return [name for name, _ in Descriptors._descList]


def compute_rdkit_descriptors(smiles: List[str], requested_names: List[str]) -> pd.DataFrame:
    """
    Compute RDKit descriptors for requested_names that exist in RDKit.
    If a name is missing in RDKit, create it as 0.0 so alignment always succeeds.
    """
    requested_names = [sanitize_column_name(x) for x in requested_names]
    requested_names = dedupe_preserve_order(requested_names)

    available = set(rdkit_available_descriptor_names())
    calc_names = [n for n in requested_names if n in available]
    missing = [n for n in requested_names if n not in available]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(calc_names) if calc_names else None

    rows_calc = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None or calc is None:
            rows_calc.append([np.nan] * len(calc_names))
            continue
        try:
            rows_calc.append(list(calc.CalcDescriptors(mol)))
        except Exception:
            rows_calc.append([np.nan] * len(calc_names))

    X = pd.DataFrame(rows_calc, columns=calc_names)
    for m in missing:
        X[m] = 0.0

    X = X[requested_names]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def load_fold_model(
    fold_dir: Path,
    device: str,
    n_features: int,
    n_hid_lay: int,
    neurons: int,
    dropout: float,
):
    """
    Prefer entire_model.pt; fallback to checkpoint.pt (state_dict).
    """
    final_model_dir = fold_dir / "Final_model"
    entire = final_model_dir / "entire_model.pt"
    ckpt = final_model_dir / "checkpoint.pt"

    if entire.exists():
        model = torch.load(entire, map_location=device, weights_only=False)
        model.eval()
        return model

    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model = Net(n_features, n_hid_lay=n_hid_lay, neurons=neurons, dropout=dropout)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    raise FileNotFoundError(f"Could not find entire_model.pt or checkpoint.pt under: {final_model_dir}")


@torch.no_grad()
def predict_proba_class1(model, X: np.ndarray, device: str) -> np.ndarray:
    """Return p(class=1) for each row (softmax(logits)[:,1])."""
    model.to(device)
    n = X.shape[0]
    p1 = np.zeros(n, dtype=float)

    for start in range(0, n, BATCH_SIZE):
        end = min(n, start + BATCH_SIZE)
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        logits = model(xb)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        p1[start:end] = probs[:, 1].detach().cpu().numpy()

    return p1


def vote_entropy_from_vote_frac(vf: float) -> float:
    """Binary vote entropy (log2)."""
    if not np.isfinite(vf):
        return np.nan
    vf = float(vf)
    if vf <= 0.0 or vf >= 1.0:
        return 0.0
    return float(-(vf * np.log2(vf) + (1 - vf) * np.log2(1 - vf)))


def summarize_distribution(p1: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(p1)
    if not mask.any():
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "entropy_mean": np.nan}

    p1v = p1[mask].astype(float)
    sd = float(np.std(p1v, ddof=1)) if len(p1v) > 1 else 0.0

    eps = 1e-12
    p = np.clip(p1v, eps, 1 - eps)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    return {
        "mean": float(np.mean(p1v)),
        "sd": sd,
        "median": float(np.median(p1v)),
        "p05": float(np.quantile(p1v, 0.05)) if len(p1v) > 1 else float(p1v[0]),
        "p95": float(np.quantile(p1v, 0.95)) if len(p1v) > 1 else float(p1v[0]),
        "entropy_mean": float(np.mean(ent)),
    }


def assign_confidence_from_votes(votes_positive: int, votes_total: int) -> str:
    """
    Vote-based confidence rules:

      8–10  -> STRONG (positive)
      6–7   -> WEAK
      5     -> UNCERTAIN
      0–2   -> STRONG (negative)
      3–4   -> WEAK
    """

    if votes_total <= 0:
        return "NA"

    # Strong positive
    if votes_positive >= 8:
        return "STRONG"

    # Strong negative
    if votes_positive <= 2:
        return "STRONG"

    # Perfect split
    if votes_positive == 5:
        return "UNCERTAIN"

    # 6–7 or 3–4
    return "WEAK"

def assign_triage_tier_hybrid(
    consensus_pred: int,
    votes_positive: int,
    votes_total: int,
    median_p1: float,
    pos_median_thr: float = 0.60,
    neg_median_thr: float = 0.40,
) -> str:
    if votes_total <= 0 or not np.isfinite(median_p1):
        return "NA"

    # Positive consensus path
    if consensus_pred == 1:
        if votes_positive >= 8 and median_p1 >= pos_median_thr:
            return "High concern"
        return "Borderline"

    # Negative consensus path (now neg_median_thr actually matters)
    # "No concern" only if strongly negative
    if votes_positive <= 2 and median_p1 <= neg_median_thr:
        return "No concern"

    return "Borderline"



class ApplicabilityDomainChecker:
    """
    Applicability-domain checker in scaled descriptor space using Euclidean distance
    against a reference pre-training set.

    Decision rule:
      - find top_k nearest reference compounds (k=25)
      - compute mean Euclidean distance across those neighbors
      - if mean distance <= threshold => inlier
        else => outlier
    """

    def __init__(self, top_k: int = 25, threshold: float = 2.3013, metric: str = "euclidean"):
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.metric = str(metric)

    def compute_distances(self, X_query: np.ndarray, X_reference: np.ndarray) -> np.ndarray:
        if X_query.ndim != 2 or X_reference.ndim != 2:
            raise ValueError("X_query and X_reference must both be 2D arrays")
        if X_query.shape[1] != X_reference.shape[1]:
            raise ValueError(
                f"Feature mismatch: query has {X_query.shape[1]} cols but reference has {X_reference.shape[1]} cols"
            )
        if X_reference.shape[0] == 0:
            raise ValueError("Reference set is empty; cannot compute applicability domain.")
        return pairwise_distances(X_query, X_reference, metric=self.metric)

    def score(self, X_query: np.ndarray, X_reference: np.ndarray) -> pd.DataFrame:
        dmat = self.compute_distances(X_query, X_reference)
        k = min(self.top_k, X_reference.shape[0])

        mean_topk = np.zeros(dmat.shape[0], dtype=float)
        min_dist = np.zeros(dmat.shape[0], dtype=float)
        max_topk = np.zeros(dmat.shape[0], dtype=float)

        for i, row in enumerate(dmat):
            topk = np.sort(row)[:k]
            mean_topk[i] = float(np.mean(topk))
            min_dist[i] = float(topk[0])
            max_topk[i] = float(topk[-1])

        labels = np.where(mean_topk <= self.threshold, "inlier", "outlier")

        return pd.DataFrame({
            "ad_mean_topk_distance": mean_topk,
            "ad_min_distance": min_dist,
            "ad_max_topk_distance": max_topk,
            "ad_label": labels,
        })


def aggregate_ad_labels(label_matrix: np.ndarray) -> List[str]:
    """Majority vote aggregation across folds for AD labels."""
    if label_matrix.ndim != 2:
        raise ValueError("label_matrix must be 2D: (n_samples, n_folds)")

    out = []
    for row in label_matrix:
        row = np.asarray(row, dtype=object)
        inlier_votes = int(np.sum(row == "inlier"))
        out.append("inlier" if inlier_votes >= (len(row) / 2.0) else "outlier")
    return out


def aggregate_ad_numeric(values: np.ndarray) -> np.ndarray:
    """Mean aggregation across folds for numeric AD diagnostics."""
    if values.ndim != 2:
        raise ValueError("values must be 2D: (n_samples, n_folds)")
    return np.nanmean(values.astype(float), axis=1)


# -------------------------
# Main
# -------------------------
def main():
    set_reproducible_inference(1234)
    
    ap = argparse.ArgumentParser(description="Forward prediction using RDKit descriptors with a CV ensemble.")
    ap.add_argument("--models_root", required=True, type=str,
                    help="Root with CV_fold_1..CV_fold_K and Descriptor_names/")
    ap.add_argument("--input_csv", required=True, type=str,
                    help="CSV containing a SMILES column (recommended name: SMILES).")
    ap.add_argument("--output_folder", required=True, type=str,
                    help="Folder path where output.csv will be written.")
    ap.add_argument("--device", default="cpu", type=str, help="cpu or cuda (e.g., cuda:0)")
    ap.add_argument("--n_folds", default=10, type=int, help="Number of CV folds to use (default 10).")

    # Net architecture defaults (must match training if using checkpoint.pt fallback)
    ap.add_argument("--n_hid_lay", default=1, type=int)
    ap.add_argument("--neurons", default=256, type=int)
    ap.add_argument("--dropout", default=0.2, type=float)

    # Consensus threshold (for consensus_pred only)
    ap.add_argument("--consensus_threshold", default=0.5, type=float,
                    help="Threshold on mean p(class1) for consensus_pred (default 0.5).")

    # Hybrid triage thresholds
    ap.add_argument("--triage_pos_median", default=0.60, type=float,
                    help="High concern requires median_p1 >= this when votes_positive>=8 (default 0.60).")
    ap.add_argument("--triage_neg_median", default=0.40, type=float,
                    help="Low concern requires median_p1 <= this when votes_positive<=2 (default 0.40).")

    # Applicability domain (AD)
    ap.add_argument("--ad_check", action="store_true",
                    help="Enable applicability domain checking against a reference pre-train set.")
    ap.add_argument("--pretrain_csv", type=str, default=None,
                    help="CSV containing reference pre-train SMILES for AD. Required when --ad_check is set.")
    ap.add_argument("--ad_top_k", default=25, type=int,
                    help="Number of nearest pre-train compounds to average for AD (default 25).")
    ap.add_argument("--ad_threshold", default=2.3013, type=float,
                    help="AD threshold on mean top-k Euclidean distance in scaled space (default 2.3013).")

    args = ap.parse_args()

    models_root = Path(args.models_root)
    device = args.device
    n_folds = int(args.n_folds)

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.csv"

    # Load input
    in_df = pd.read_csv(args.input_csv)
    smi_col = infer_smiles_column(in_df)
    smiles_list = in_df[smi_col].astype(str).tolist()

    # Optional AD reference set
    pretrain_smiles: List[str] = []
    if args.ad_check:
        if not args.pretrain_csv:
            raise ValueError("--pretrain_csv is required when --ad_check is enabled.")
        pretrain_df = pd.read_csv(args.pretrain_csv)
        pretrain_smi_col = infer_smiles_column(pretrain_df)
        pretrain_smiles = pretrain_df[pretrain_smi_col].astype(str).tolist()
        if len(pretrain_smiles) == 0:
            raise ValueError("Pre-train CSV is empty; cannot perform AD checking.")

    # Base descriptor list from training
    base_desc_path = models_root / "Descriptor_names" / "Descr_names_non_inf.csv"
    if not base_desc_path.exists():
        raise FileNotFoundError(f"Missing base descriptor list: {base_desc_path}")
    base_cols = load_noheader_list_csv(base_desc_path)

    # Compute descriptors for base columns
    X_base = compute_rdkit_descriptors(smiles_list, base_cols)
    X_base.columns = [sanitize_column_name(c) for c in X_base.columns]

    X_pretrain_base = None
    if args.ad_check:
        X_pretrain_base = compute_rdkit_descriptors(pretrain_smiles, base_cols)
        X_pretrain_base.columns = [sanitize_column_name(c) for c in X_pretrain_base.columns]

    n_rows = X_base.shape[0]
    p1_stack = np.full((n_rows, n_folds), np.nan, dtype=float)
    pred_stack = np.full((n_rows, n_folds), np.nan, dtype=float)

    ad_label_stack = np.full((n_rows, n_folds), "NA", dtype=object) if args.ad_check else None
    ad_mean_stack = np.full((n_rows, n_folds), np.nan, dtype=float) if args.ad_check else None
    ad_min_stack = np.full((n_rows, n_folds), np.nan, dtype=float) if args.ad_check else None
    ad_max_topk_stack = np.full((n_rows, n_folds), np.nan, dtype=float) if args.ad_check else None

    ad_checker = ApplicabilityDomainChecker(
        top_k=int(args.ad_top_k),
        threshold=float(args.ad_threshold),
        metric="euclidean",
    ) if args.ad_check else None

    for fold in range(1, n_folds + 1):
        fold_dir = models_root / f"CV_fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")

        # Load scaler
        scaler_path = fold_dir / "scaler" / "MinMaxScaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Determine scaler expected columns
        descr_for_scaling_path = fold_dir / "scaler" / "Descr_for_scaling.csv"
        if descr_for_scaling_path.exists():
            expected_cols = load_noheader_list_csv(descr_for_scaling_path)
        elif hasattr(scaler, "feature_names_in_"):
            expected_cols = [sanitize_column_name(c) for c in scaler.feature_names_in_.tolist()]
            expected_cols = dedupe_preserve_order(expected_cols)
        else:
            expected_cols = base_cols

        # Align to expected scaler columns
        X_fold = X_base.copy()
        for c in expected_cols:
            if c not in X_fold.columns:
                X_fold[c] = 0.0
        X_fold = X_fold[expected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Scale
        X_scaled = scaler.transform(X_fold.values)

        X_pre_scaled = None
        if args.ad_check:
            X_pre = X_pretrain_base.copy()
            for c in expected_cols:
                if c not in X_pre.columns:
                    X_pre[c] = 0.0
            X_pre = X_pre[expected_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_pre_scaled = scaler.transform(X_pre.values)

        # Apply variance-selected columns
        vt_sel_path = fold_dir / "var_threshold_select_descr" / "Selected_desc_variance_thre.csv"
        if not vt_sel_path.exists():
            raise FileNotFoundError(f"Missing VT-selected descriptor list: {vt_sel_path}")
        sel_cols = load_noheader_list_csv(vt_sel_path)

        col_to_idx = {c: j for j, c in enumerate(expected_cols)}
        sel_idx = [col_to_idx[c] for c in sel_cols if c in col_to_idx]
        if len(sel_idx) == 0:
            raise ValueError(f"Fold {fold}: no selected columns matched scaler columns.")

        X_sel = X_scaled[:, sel_idx]
        X_pre_sel = X_pre_scaled[:, sel_idx] if args.ad_check else None
        n_features = X_sel.shape[1]

        if args.ad_check:
            ad_scores = ad_checker.score(X_query=X_sel, X_reference=X_pre_sel)
            ad_label_stack[:, fold - 1] = ad_scores["ad_label"].to_numpy(dtype=object)
            ad_mean_stack[:, fold - 1] = ad_scores["ad_mean_topk_distance"].to_numpy(dtype=float)
            ad_min_stack[:, fold - 1] = ad_scores["ad_min_distance"].to_numpy(dtype=float)
            ad_max_topk_stack[:, fold - 1] = ad_scores["ad_max_topk_distance"].to_numpy(dtype=float)

        # Load model
        model = load_fold_model(
            fold_dir=fold_dir,
            device=device,
            n_features=n_features,
            n_hid_lay=int(args.n_hid_lay),
            neurons=int(args.neurons),
            dropout=float(args.dropout),
        )

        # Predict fold probabilities
        p1 = predict_proba_class1(model, X_sel, device=device)
        yhat = (p1 >= 0.5).astype(int)

        p1_stack[:, fold - 1] = p1
        pred_stack[:, fold - 1] = yhat

    # Aggregate
    CONS_THR = float(args.consensus_threshold)
    pos_med_thr = float(args.triage_pos_median)
    neg_med_thr = float(args.triage_neg_median)

    ad_final = aggregate_ad_labels(ad_label_stack) if args.ad_check else ["NA"] * n_rows
    ad_mean_final = aggregate_ad_numeric(ad_mean_stack) if args.ad_check else np.full(n_rows, np.nan, dtype=float)
    ad_min_final = aggregate_ad_numeric(ad_min_stack) if args.ad_check else np.full(n_rows, np.nan, dtype=float)
    ad_max_topk_final = aggregate_ad_numeric(ad_max_topk_stack) if args.ad_check else np.full(n_rows, np.nan, dtype=float)

    rows: List[Dict[str, Any]] = []

    for i in range(n_rows):
        p1_row = p1_stack[i, :]
        pred_row = pred_stack[i, :]

        votes_total = int(np.sum(np.isfinite(pred_row)))
        votes_positive = int(np.nansum(pred_row == 1)) if votes_total > 0 else 0
        vote_frac = (votes_positive / votes_total) if votes_total > 0 else np.nan
        vote_entropy = vote_entropy_from_vote_frac(vote_frac)

        dist = summarize_distribution(p1_row)
        mean_p1 = dist["mean"]
        median_p1 = dist["median"]

        consensus_pred = int(mean_p1 >= CONS_THR) if np.isfinite(mean_p1) else np.nan

        confidence = assign_confidence_from_votes(votes_positive=votes_positive, votes_total=votes_total)

        triage_tier = assign_triage_tier_hybrid(
            consensus_pred=consensus_pred,
            votes_positive=votes_positive,
            votes_total=votes_total,
            median_p1=median_p1,
            pos_median_thr=pos_med_thr,
            neg_median_thr=neg_med_thr,
        )
        
        rows.append({
            "row_id": i,
            "SMILES": smiles_list[i],
            "consensus_pred": consensus_pred,
            "votes_positive": votes_positive,
            "votes_total": votes_total,
            "vote_frac": vote_frac,
            "vote_entropy": vote_entropy,
            "confidence": confidence,
            "triage_tier": triage_tier,
            "applicability_domain": ad_final[i],
            "ad_mean_topk_distance": ad_mean_final[i],
            "ad_min_distance": ad_min_final[i],
            "ad_max_topk_distance": ad_max_topk_final[i],
        })

    out_df = pd.DataFrame(rows, columns=[
        "row_id", "SMILES", "consensus_pred",
        "votes_positive", "votes_total", "vote_frac", "vote_entropy",
        "confidence", "triage_tier",
        "applicability_domain", "ad_mean_topk_distance", "ad_min_distance", "ad_max_topk_distance"
    ])
    out_df.to_csv(out_path, index=False)

    print("\nDONE.")
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
