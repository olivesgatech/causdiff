import os
import glob
import csv
import numpy as np
from typing import Dict, List, Tuple

_EPS = 1e-12

# ---------------------------
# Utils
# ---------------------------

def _safe_nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    x = np.where(np.isfinite(x), x, np.nan)
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))

def _sanitize_probs(P: np.ndarray) -> np.ndarray:
    """Ensure valid probs (S,T,C): non-neg, finite, sum=1 on last axis."""
    P = np.asarray(P, dtype=float)
    P = np.where(np.isfinite(P), P, 0.0)
    P = np.clip(P, 0.0, None)
    Z = P.sum(axis=-1, keepdims=True)
    Z = np.where(Z > 0.0, Z, 1.0)
    return P / Z

def softmax_last(x: np.ndarray, from_logits: bool = True) -> np.ndarray:
    if not from_logits:
        return _sanitize_probs(x)
    m = np.max(x, axis=-1, keepdims=True)
    z = np.exp(x - m)
    z_sum = np.sum(z, axis=-1, keepdims=True)
    return _sanitize_probs(z / np.clip(z_sum, _EPS, None))

def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

# ---------------------------
# Between-sample metrics
# ---------------------------

def mutual_information_from_samples(P: np.ndarray) -> np.ndarray:
    """MI_t = H(mean_s P_t) - mean_s H(P_{s,t}); shape (T,)"""
    if P.ndim != 3 or P.shape[1] == 0 or P.shape[2] == 0:
        return np.array([], dtype=float)
    P = _sanitize_probs(P)
    P_bar = np.mean(P, axis=0)                  # (T,C)
    H_t = entropy(P_bar, axis=-1)               # (T,)
    Htilde_t = np.mean(entropy(P, axis=-1), axis=0)  # (T,)
    MI_t = H_t - Htilde_t
    return np.where(np.isfinite(MI_t), MI_t, np.nan)

def aleatoric_expected_entropy(P: np.ndarray) -> np.ndarray:
    """Average per-sample entropy at each t (aleatoric proxy)."""
    P = _sanitize_probs(P)
    return np.mean(entropy(P, axis=-1), axis=0)  # (T,)

def disagreement_rate(P: np.ndarray) -> np.ndarray:
    """Frac of samples whose argmax != ensemble argmax at each t."""
    if P.ndim != 3 or P.shape[0] < 2:
        return np.array([], dtype=float)
    P = _sanitize_probs(P)
    y_bar = np.argmax(np.mean(P, axis=0), axis=-1)  # (T,)
    y_s = np.argmax(P, axis=-1)                     # (S,T)
    disagree = np.mean(y_s != y_bar[None, :], axis=0)
    return np.where(np.isfinite(disagree), disagree, np.nan)

def pairwise_hamming_diversity(P: np.ndarray) -> float:
    """Mean pairwise Hamming distance of argmax paths (normalized by T)."""
    P = _sanitize_probs(P)
    S, T, C = P.shape
    if S < 2:
        return np.nan
    y = np.argmax(P, axis=-1)  # (S,T)
    total = 0.0; cnt = 0
    for i in range(S):
        for j in range(i+1, S):
            total += np.mean(y[i] != y[j])
            cnt += 1
    return float(total / max(cnt,1))

def pairwise_l2_diversity(P: np.ndarray) -> float:
    """Mean pairwise L2 distance between flattened probability sequences."""
    P = _sanitize_probs(P)
    S, T, C = P.shape
    if S < 2:
        return np.nan
    X = P.reshape(S, T*C)
    total = 0.0; cnt = 0
    for i in range(S):
        for j in range(i+1, S):
            diff = X[i] - X[j]
            total += np.linalg.norm(diff)
            cnt += 1
    return float(total / max(cnt,1))

def between_sample_variance(P: np.ndarray) -> np.ndarray:
    """Sum_c Var_s[P[s,t,c]] per t (epistemic proxy via variance)."""
    P = _sanitize_probs(P)
    return np.sum(np.var(P, axis=0), axis=-1)  # (T,)

def temporal_jitter(P: np.ndarray) -> Dict[str, float]:
    """
    Dispersion of change times and number of changes across samples.
    """
    P = _sanitize_probs(P)
    S, T, C = P.shape
    y = np.argmax(P, axis=-1)  # (S,T)
    first_changes, change_counts = [], []
    for s in range(S):
        ys = y[s]
        changes = np.nonzero(ys[1:] != ys[:-1])[0] + 1
        change_counts.append(len(changes))
        if len(changes) > 0:
            first_changes.append(changes[0])
    return {
        "first_change_std": _safe_nanmean(first_changes) if len(first_changes) >= 2 else np.nan,
        "changes_std": np.std(change_counts) if S >= 2 else np.nan,
    }

# ---------------------------
# Multi-modality across samples (cluster S samples)
# ---------------------------

def _pca_reduce(X: np.ndarray, dmax: int = 64) -> np.ndarray:
    """
    X: (S, Dflat). Returns PCA-reduced features (S, d) with d<=dmax and d<=rank.
    """
    S, D = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    U, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = int(min(dmax, len(Svals)))
    if d == 0:
        return Xc
    return (U[:, :d] * Svals[:d])  # (S,d)

def _gmm_diag_em_samples(X: np.ndarray, K: int, n_iter: int = 50, seed: int = 0):
    """
    GMM over samples (S x d). Very small diagonal-cov EM to estimate modes.
    """
    rng = np.random.RandomState(seed)
    S, d = X.shape
    idx = rng.randint(0, K, size=S)
    mu = np.array([X[idx==k].mean(axis=0) if np.any(idx==k) else X.mean(axis=0) for k in range(K)])
    var = np.array([X[idx==k].var(axis=0) + 1e-6 if np.any(idx==k) else X.var(axis=0) + 1e-6 for k in range(K)])
    pi = np.full(K, 1.0/K)
    for _ in range(n_iter):
        log_resp = np.zeros((S,K))
        for k in range(K):
            diff = X - mu[k]
            invvar = 1.0 / (var[k] + 1e-6)
            log_det = -0.5*np.sum(np.log(2*np.pi*(var[k]+1e-6)))
            log_prob = log_det - 0.5*np.sum(diff*diff*invvar, axis=1)
            log_resp[:,k] = np.log(pi[k] + _EPS) + log_prob
        m = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - m)
        resp = resp / np.clip(resp.sum(axis=1, keepdims=True), _EPS, None)
        Nk = resp.sum(axis=0) + _EPS
        pi = Nk / S
        mu = (resp.T @ X) / Nk[:,None]
        var = (resp.T @ (X**2)) / Nk[:,None] - mu**2
        var = np.maximum(var, 1e-6)
    # final loglike
    ll = 0.0
    for i in range(S):
        vals = []
        for k in range(K):
            diff = X[i] - mu[k]
            invvar = 1.0 / (var[k] + 1e-6)
            log_det = -0.5*np.sum(np.log(2*np.pi*(var[k]+1e-6)))
            vals.append(np.log(pi[k] + _EPS) + log_det - 0.5*np.sum(diff*diff*invvar))
        m = np.max(vals)
        ll += m + np.log(np.sum(np.exp(np.array(vals)-m)))
    return pi, mu, var, ll

def _bic(ll: float, K: int, d: int, S: int) -> float:
    n_params = (K-1) + K*d + K*d
    return -2*ll + n_params*np.log(max(S,1))

def multimodality_across_samples(P: np.ndarray, Kmax: int = 6, seeds: List[int] = [0,1,2]) -> Dict[str, float]:
    """
    Cluster S samples into K modes using PCA-reduced flattened probabilities.
    """
    P = _sanitize_probs(P)            # (S,T,C)
    S, T, C = P.shape
    if S < 2:
        return {"mmK": np.nan, "mm_balance": np.nan, "mm_sep": np.nan}
    X = P.reshape(S, T*C)
    Xr = _pca_reduce(X, dmax=64)      # (S,d)
    best = {"BIC": np.inf, "K": 1, "pi": np.array([1.0]), "mu": None}
    for K in range(1, Kmax+1):
        best_ll = -np.inf; best_params=None
        for sd in seeds:
            pi, mu, var, ll = _gmm_diag_em_samples(Xr, K, seed=sd)
            if ll > best_ll:
                best_ll = ll; best_params=(pi, mu)
        bic = _bic(best_ll, K, Xr.shape[1], S)
        if bic < best["BIC"]:
            best = {"BIC": bic, "K": K, "pi": best_params[0], "mu": best_params[1]}
    pi = np.clip(best["pi"], _EPS, 1.0)
    balance = -np.sum(pi*np.log(pi))/np.log(len(pi))
    mu = best["mu"]
    if mu is None or mu.shape[0] == 1:
        sep = 0.0
    else:
        diffs = []
        for i in range(mu.shape[0]):
            for j in range(i+1, mu.shape[0]):
                diffs.append(np.linalg.norm(mu[i]-mu[j]))
        sep = float(np.mean(diffs)) if diffs else 0.0
    return {"mmK": float(best["K"]), "mm_balance": float(balance), "mm_sep": sep}

# ---------------------------
# One-shot ensemble evaluator
# ---------------------------

def evaluate_ensemble_uncertainties(P_group: np.ndarray) -> Dict[str, float]:
    """
    P_group: (S, T, C) probabilities.
    Returns ensemble-level metrics that compare between samples.
    """
    if P_group.ndim != 3:
        raise ValueError(f"Expected (S,T,C), got {P_group.shape}")
    S, T, C = P_group.shape
    P_group = _sanitize_probs(P_group)

    # Epistemic vs Aleatoric
    MI_t = mutual_information_from_samples(P_group)       # (T,)
    Htilde_t = aleatoric_expected_entropy(P_group)        # (T,)
    var_t = between_sample_variance(P_group)              # (T,)

    # Disagreement / diversity
    disagree_t = disagreement_rate(P_group)               # (T,) or []
    ham = pairwise_hamming_diversity(P_group)             # scalar
    l2div = pairwise_l2_diversity(P_group)                # scalar

    # Temporal jitter across samples
    tj = temporal_jitter(P_group)

    return {
        "MI_mean": _safe_nanmean(MI_t),
        "Aleatoric_meanH": _safe_nanmean(Htilde_t),
        "Var_between_mean": _safe_nanmean(var_t),
        "Disagree_mean": _safe_nanmean(disagree_t),
        "Pairwise_Hamming": ham,
        "Pairwise_L2": l2div,
        "FirstChange_STD": tj["first_change_std"],
        "NumChanges_STD": tj["changes_std"],
    }

# ---------------------------
# I/O and comparison
# ---------------------------

def _load(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"{os.path.basename(path)}: expected 4D (S,1,T,C), got {arr.shape}")
    return arr  # (S,1,T,C)

def compare_three_folders(
    folder_A: str,
    folder_AG: str,
    folder_AS: str,
    save_csv_path: str,
    from_logits: bool = True,
    Kmax_modes: int = 6
) -> Dict[str, Dict[str, float]]:
    """
    For each common filename:
      - Load (S,1,T,C), convert to probs
      - Compute ensemble-level metrics across S (between-sample)
      - Also compute multi-modality across samples via clustering
      - Write one CSV row per (filename, condition)
      - Return overall means per condition
    """
    files_A  = {os.path.basename(p): p for p in glob.glob(os.path.join(folder_A, "*.npy"))}
    files_AG = {os.path.basename(p): p for p in glob.glob(os.path.join(folder_AG, "*.npy"))}
    files_AS = {os.path.basename(p): p for p in glob.glob(os.path.join(folder_AS, "*.npy"))}

    common = sorted(set(files_A) & set(files_AG) & set(files_AS))
    if not common:
        raise FileNotFoundError("No common .npy filenames across the three folders.")

    os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
    fieldnames = [
        "filename", "condition",
        "MI_mean", "Aleatoric_meanH", "Var_between_mean",
        "Disagree_mean", "Pairwise_Hamming", "Pairwise_L2",
        "FirstChange_STD", "NumChanges_STD",
        "mmK_samples", "mm_balance_samples", "mm_sep_samples"
    ]

    rows = []
    agg = {"A": [], "AG": [], "AS": []}

    for fname in common:
        A_arr  = _load(files_A[fname])   # (S,1,T,C)
        AG_arr = _load(files_AG[fname])
        AS_arr = _load(files_AS[fname])

        # Align S across conditions (just in case)
        Smin = min(A_arr.shape[0], AG_arr.shape[0], AS_arr.shape[0])
        A_arr, AG_arr, AS_arr = A_arr[:Smin], AG_arr[:Smin], AS_arr[:Smin]

        # Convert to probabilities
        A_prob  = softmax_last(A_arr[:,0,:,:], from_logits=from_logits)   # (S,T,C)
        AG_prob = softmax_last(AG_arr[:,0,:,:], from_logits=from_logits)
        AS_prob = softmax_last(AS_arr[:,0,:,:], from_logits=from_logits)

        for cond, P in [("A", A_prob), ("AG", AG_prob), ("AS", AS_prob)]:
            ens = evaluate_ensemble_uncertainties(P)
            mm  = multimodality_across_samples(P, Kmax=Kmax_modes)
            row = [
                fname, cond,
                ens["MI_mean"], ens["Aleatoric_meanH"], ens["Var_between_mean"],
                ens["Disagree_mean"], ens["Pairwise_Hamming"], ens["Pairwise_L2"],
                ens["FirstChange_STD"], ens["NumChanges_STD"],
                mm["mmK"], mm["mm_balance"], mm["mm_sep"]
            ]
            rows.append(row)
            agg[cond].append({**ens, **{"mmK_samples": mm["mmK"], "mm_balance_samples": mm["mm_balance"], "mm_sep_samples": mm["mm_sep"]}})

    # Save CSV
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    # Aggregate means per condition
    def _mean_dict(list_of_dicts: List[Dict[str,float]]) -> Dict[str, float]:
        keys = list_of_dicts[0].keys()
        out = {}
        for k in keys:
            vals = np.array([d[k] for d in list_of_dicts], dtype=float)
            vals = np.where(np.isfinite(vals), vals, np.nan)
            out[k] = _safe_nanmean(vals)
        return out

    means = {cond: _mean_dict(agg[cond]) for cond in ["A","AG","AS"]}
    print(f"Compared {len(common)} common files. Saved CSV to: {save_csv_path}")
    return means

# ---------------------------
# Example usage (edit paths)
# ---------------------------
# means = compare_three_folders(
#     folder_A="/mnt/data-tmp/seulgi/causdiff/src/output_baseline_48",
#     folder_AG="/mnt/data-tmp/seulgi/causdiff/src/output_goal_48",
#     folder_AS="/mnt/data-tmp/seulgi/causdiff/src/output_proposed_48",
#     save_csv_path="/mnt/data-tmp/seulgi/causdiff/src/uncertainty/uncertainty_between_samples_48.csv",
#     from_logits=True,   # set False if arrays already are probabilities
#     Kmax_modes=6
# )
# print(means)


means = compare_three_folders(
    folder_A="/mnt/data-tmp/seulgi/causdiff/src/output_baseline_48",
    folder_AG="/mnt/data-tmp/seulgi/causdiff/src/output_goal_48",
    folder_AS="/mnt/data-tmp/seulgi/causdiff/src/output_proposed_48",
    save_csv_path="/mnt/data-tmp/seulgi/causdiff/src/uncertainty/uncertainty_three_conditions_48.csv",
    from_logits=True,
    Kmax_modes=6
)
print(means)
