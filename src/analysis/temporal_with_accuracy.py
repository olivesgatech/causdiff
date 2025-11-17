import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional, Tuple
from please import *
from deepseek import *
from manifold import *

file_list = [
    "16_3_camera_2_fps_15_Using handheld smart devices.txt",
    "12_2_camera_2_fps_15_Using handheld smart devices.txt",
    "18_4_camera_2_fps_15_Using handheld smart devices.txt",
    "14_3_camera_2_fps_15_Using handheld smart devices.txt",
    "19_2_camera_1_fps_15_Using handheld smart devices.txt",
    "06_3_camera_1_fps_15_Using handheld smart devices.txt",
    "18_3_camera_1_fps_15_Using handheld smart devices.txt",
    "12_3_camera_1_fps_15_Making a cup of instant coffee.txt",
    "17_3_camera_1_fps_15_Making a cup of instant coffee.txt",
    "16_3_camera_1_fps_15_Making a cup of instant coffee.txt",
    "07_3_camera_2_fps_15_Making a cup of instant coffee.txt",
    "06_4_camera_2_fps_15_Making a cup of instant coffee.txt",
    "13_4_camera_2_fps_15_Making pancake.txt",
    "20_3_camera_2_fps_15_Making pancake.txt",
    "02_3_camera_2_fps_15_Making pancake.txt",
    "01_3_camera_1_fps_15_Making pancake.txt",
    "10_4_camera_1_fps_15_Making pancake.txt",
    "09_3_camera_1_fps_15_Making pancake.txt",
    "01_3_camera_2_fps_15_Making pancake.txt",
    "20_3_camera_2_fps_15_Dining.txt",
    "20_4_camera_1_fps_15_Dining.txt",
    "08_4_camera_1_fps_15_Cleaning dishes.txt",
    "15_4_camera_2_fps_15_Cleaning dishes.txt",
    "07_4_camera_1_fps_15_Cleaning dishes.txt",
    "18_3_camera_2_fps_15_Cleaning dishes.txt",
    "17_4_camera_1_fps_15_Cleaning dishes.txt",
    "15_4_camera_1_fps_15_Cleaning dishes.txt",
    "02_4_camera_1_fps_15_Cleaning dishes.txt",
    "03_3_camera_1_fps_15_Cleaning dishes.txt",
    "13_4_camera_1_fps_15_Making a cup of coffee in coffee maker.txt",
    "13_4_camera_2_fps_15_Making a cup of coffee in coffee maker.txt",
    "07_4_camera_1_fps_15_Making a cup of coffee in coffee maker.txt",
    "20_3_camera_2_fps_15_Cleaning the kitchen.txt",
    "20_3_camera_1_fps_15_Cleaning the kitchen.txt"
]

INFER_DIR = "/home/hice1/skim3513/scratch/causdiff/outputs/infer_goal/20"
MOCS_DIR  = "/home/hice1/skim3513/scratch/causdiff/outputs/mocs_20"
OUT_DIR   = "/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20"
img_path = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/features_img"
global_goal = [0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,3,3,4,4,4,4,4,4,4,4,5,5,5,6,6,]
os.makedirs(OUT_DIR, exist_ok=True)

PREFIX = "infer_goal_t0_"
SUFFIX = ".npy"

# ---------- 1) 파일 수집: npy(비디오) 타임스탬프 순 정렬 ----------
pairs = []
for f in glob.glob(os.path.join(INFER_DIR, f"{PREFIX}*{SUFFIX}")):
    base = os.path.basename(f)
    if base.startswith(PREFIX) and base.endswith(SUFFIX):
        token = base[len(PREFIX):-len(SUFFIX)]  # timestamp-like token
        pairs.append((token, f))
if not pairs:
    raise FileNotFoundError(f"No files like {PREFIX}*{SUFFIX} in {INFER_DIR}")
pairs.sort(key=lambda x: x[0])
ordered_paths = [p[1] for p in pairs]

# ---------- 2) 유틸: PR형 effective rank ----------
def pr_effective_rank_from_samples(X: np.ndarray) -> float:
    """
    X: (S, D) 샘플(동일 시각, 동일 배치)
    Participation-ratio effective rank on covariance: (tr(C))^2 / tr(C^2)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    S = X.shape[0]
    if S <= 1:
        return float("nan")
    C = (Xc.T @ Xc) / (S - 1)
    tr  = float(np.trace(C))
    tr2 = float(np.sum(C * C))
    if tr2 <= 1e-12:
        return 0.0
    return (tr * tr) / tr2

def spectral_entropy_from_samples(X, eps=1e-12):
    # entropy over eigenvalues of covariance
    Xc = X - X.mean(axis=0, keepdims=True)
    S = X.shape[0]
    if S <= 1: return float("nan")
    C = (Xc.T @ Xc) / (S - 1)
    lam = np.linalg.eigvalsh(C).clip(min=0)
    s = lam.sum() + eps
    p = lam / s
    H = float(-(p * np.log(p + eps)).sum())
    return H  # also can report effective rank = exp(H)

def mean_pairwise_distance(X):
    # simple spread proxy, O(S^2), S=5 is fine
    S = X.shape[0]
    if S < 2: return float("nan")
    dsum = 0.0
    cnt = 0
    for i in range(S):
        for j in range(i+1, S):
            dsum += np.linalg.norm(X[i]-X[j])
            cnt += 1
    return dsum / max(cnt,1)

def _center_along_samples(X_std: np.ndarray) -> np.ndarray:
    """
    X_std: (S, T, D)
    Return: centered along S for each (t, d): Xc[s,t,d] = X[s,t,d] - mean_s X[:,t,d]
    """
    return X_std - X_std.mean(axis=0, keepdims=True)

def vn_entropy_from_samples(
    X_s_d: np.ndarray,
    method: Literal["gram", "cov"] = "gram",
    eps: float = 1e-12,
    return_effective_rank: bool = True
) -> Tuple[float, Optional[float]]:
    """
    Von Neumann entropy for a *single time step* from S samples in D-dim.

    Parameters
    ----------
    X_s_d : np.ndarray
        Shape (S, D), S = #samples, D = feature dim.
    method : {"gram","cov"}
        "gram": use SxS Gram matrix G = Xc Xc^T / (S-1) (fast when S << D)
        "cov" : use DxD covariance C = Xc^T Xc / (S-1)
        Nonzero eigenspectra are identical; results are equivalent.
    eps : float
        Numerical jitter and min clamp for eigenvalues and logs.
    return_effective_rank : bool
        If True, also return exp(S_vN) as entropy-based effective rank.

    Returns
    -------
    S_vN : float
        Von Neumann entropy at this time (natural log base).
    r_eff : Optional[float]
        Entropy-based effective rank = exp(S_vN), if requested.
    """
    S, D = X_s_d.shape
    if S <= 1:
        return float("nan"), (float("nan") if return_effective_rank else None)

    Xc = X_s_d - X_s_d.mean(axis=0, keepdims=True)

    if method == "gram":
        # G = Xc Xc^T / (S-1)  (SxS)
        G = (Xc @ Xc.T) / max(S - 1, 1)
        # symmetric PSD → use eigvalsh
        lam = np.linalg.eigvalsh(G).clip(min=0.0)
    elif method == "cov":
        # C = Xc^T Xc / (S-1)  (DxD)
        C = (Xc.T @ Xc) / max(S - 1, 1)
        lam = np.linalg.eigvalsh(C).clip(min=0.0)
    else:
        raise ValueError("method must be 'gram' or 'cov'")

    # Normalize to a density matrix spectrum: p_i = λ_i / trace
    trace = float(lam.sum()) + eps
    p = lam / trace
    # Von Neumann entropy: -sum p log p
    S_vN = float(-(p * np.log(p + eps)).sum())
    r_eff = float(np.exp(S_vN)) if return_effective_rank else None
    return S_vN, r_eff

def sample_variance_uncertainty(last_vecs: np.ndarray, reduce: str = "mean") -> float:
    """
    Uncertainty = variance across samples (axis=0) of last_vecs (S, D).
    Parameters
    ----------
    last_vecs : (S, D) array
        S = #samples (diffusion samples), D = feature dim
    reduce : {"mean", "sum", "trace", "median", "none"}
        How to reduce per-dimension variance to a scalar:
        - "mean": average variance across D (scale-invariant across D)
        - "sum"/"trace": sum of variances across D (== trace of covariance)
        - "median": median variance across D (robust)
        - "none": return the full (D,) variance vector

    Returns
    -------
    float or np.ndarray
        Scalar uncertainty (default) or (D,) vector if reduce="none".
    """
    assert last_vecs.ndim == 2, "last_vecs must be (S, D)"
    # ddof=1 → unbiased sample variance
    var_d = np.var(last_vecs, axis=0, ddof=1)

    if reduce == "mean":
        return float(np.nanmean(var_d))
    elif reduce in ("sum", "trace"):
        return float(np.nansum(var_d))         # equals trace of sample covariance
    elif reduce == "median":
        return float(np.nanmedian(var_d))
    elif reduce == "none":
        return var_d
    else:
        raise ValueError("reduce must be one of {'mean','sum','trace','median','none'}")

# ---------- 3) 정확도 파서: 모든 샘플 읽기 ----------
_LINE_RE = re.compile(
    r"batch\s*:\s*(?P<bid>\d+)\s*.*?accuracy\s*:\s*(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

def read_accuracies_all_samples(video_id: int, eval_len: int, S: int = 5) -> np.ndarray:
    """
    'moc_{video_id}_{eval_len}.txt'에서 sample(bid=0..S-1) accuracy 모두 읽어 길이 S 배열 반환.
    누락 샘플은 NaN.
    """
    path = os.path.join(MOCS_DIR, f"moc_{video_id}_{eval_len}.txt")
    acc = np.full(S, np.nan, dtype=float)
    if not os.path.exists(path):
        return acc

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = _LINE_RE.search(line)
            if m:
                try:
                    bid = int(m.group("bid"))
                    aval = float(m.group("acc"))
                    if 0 <= bid < S:
                        acc[bid] = aval
                except Exception:
                    pass
            else:
                # 백업 파서
                parts = [p for p in line.split("\t") if ":" in p]
                kv = {}
                for p in parts:
                    k, v = p.split(":", 1)
                    kv[k.strip().lower()] = v.strip()
                if "batch" in kv and "accuracy" in kv:
                    try:
                        bid = int(kv["batch"])
                        aval = float(kv["accuracy"])
                        if 0 <= bid < S:
                            acc[bid] = aval
                    except Exception:
                        pass
    return acc

def plot_uncertainties(X, g_schedule=None, out_prefix="/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20/uncertainty"):
    """
    Compute and plot several time-varying uncertainties.
    Saves one PNG per chart and returns the file paths.
    """
    files = []
    # Core metrics
    U_vN, mts = temporal_uncertainty_U(X, metric="vN", g_schedule=g_schedule)
    U_logdet, _ = temporal_uncertainty_U(X, metric="logdet", g_schedule=g_schedule)
    U_trace, _  = temporal_uncertainty_U(X, metric="trace", g_schedule=g_schedule)
    U_fisher, _ = temporal_uncertainty_U(X, metric="fisher", g_schedule=g_schedule)
    U_PI, _     = temporal_uncertainty_U(X, metric="PI", g_schedule=g_schedule)

    t = np.arange(U_vN.shape[0])

    # Each chart in its own figure (per tool requirement)
    for name, series in [
        ("U_vN", U_vN),
        ("U_logdet", U_logdet),
        ("U_trace", U_trace),
        ("U_fisher", U_fisher),
        ("U_PI", U_PI),
        ("vN_raw", mts["vN"]),
        ("logdet_raw", mts["logdet"]),
        ("trace_raw", mts["trace"]),
        ("fisher_proxy_raw", mts["fisher_proxy"]),
        ("PI_norm_width_raw", mts["PI_norm_width"]),
    ]:
        plt.figure()
        plt.plot(t, series)
        plt.xlabel("time t")
        plt.ylabel(name)
        plt.title(name)
        fname = f"{out_prefix}_{name}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        plt.close()
        files.append(fname)

    # Also save a tidy CSV for inspection
    df = pd.DataFrame({
        "t": t,
        "U_vN": U_vN, "U_logdet": U_logdet, "U_trace": U_trace, "U_fisher": U_fisher, "U_PI": U_PI,
        "vN": mts["vN"], "logdet": mts["logdet"], "trace": mts["trace"],
        "fisher_proxy": mts["fisher_proxy"], "PI_norm_width": mts["PI_norm_width"],
    })
    csv_path = f"{out_prefix}_timeseries.csv"
    df.to_csv(csv_path, index=False)
    files.append(csv_path)
    return files

def plot(uncertainty, out_prefix="/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20/uncertainty"):
    T = uncertainty.shape
    T = T[0]
    # 시각화
    plt.figure(figsize=(12, 4))
    plt.plot(range(T), uncertainty, 'b-', linewidth=2)
    plt.xlabel('video time τ')
    plt.ylabel('Temporal Uncertainty')
    plt.title('uncertainty based on video time sequence')
    plt.grid(True)

    # 정보 획득/손실 구간 표시
    plt.axvspan(10, 20, alpha=0.2, color='red', label='information loss')
    plt.axvspan(30, 40, alpha=0.2, color='green', label='information acquired')
    plt.legend()
    fname = f"{out_prefix}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()

# ---------- 4) per-video 레코드 생성 (mean/max 동시 수집) ----------
records = []
visual_features = load_visual_features(file_list, img_path)
idx = 0
all_samples_batch = []
intention_labels_batch = []
for vid, npy_path in enumerate(ordered_paths):
    arr = np.load(npy_path, allow_pickle=False)  # (S, B, T, D)
    if arr.ndim != 4:
        raise ValueError(f"Unexpected shape in {npy_path}: {arr.shape}")
    S, B, T, D = arr.shape
    if B < 1:
        raise ValueError(f"No batch dim in {npy_path}")

    # batch 0만 사용
    X = arr[:, 0, :, :]      # (S, T, D)
    all_samples, intention_labels = analyze_manifold_structure(X, global_goal)
    all_samples_batch.extend(all_samples)
    intention_labels_batch.extend(intention_labels)

    
    uncertainty = compute_intrinsic_uncertainty(X)
    analyze_uncertainty_correlations(uncertainty, visual_features[idx])
    
    U_vN, _ = temporal_uncertainty_U(X, metric="vN", g_schedule=None)
    uncertainty['u_vn'] = U_vN
    plot_uncertainty_analysis(uncertainty, visual_features[idx], out_prefix="/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20/uncertainty/"+str(idx))
    # temporal_uncertainty = np.var(X, axis=0)  # (T, D)
    # temporal_uncertainty = np.mean(temporal_uncertainty, axis=-1)  # (T)
    
    #out_files = plot_uncertainties(X, g_schedule=None, out_prefix="/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20/uncertainty/"+str(idx))
    #plot(temporal_uncertainty, out_prefix="/home/hice1/skim3513/scratch/causdiff/outputs/analysis_20/uncertainty/"+str(idx))
    idx += 1
    #last_vecs = X[:, -1, :]  # (S, D)
    #last_unc = pr_effective_rank_from_samples(last_vecs)
    #last_unc = spectral_entropy_from_samples(last_vecs)
    #last_unc = mean_pairwise_distance(last_vecs)
    #last_unc, _ = vn_entropy_from_samples(last_vecs)
    #last_unc = sample_variance_uncertainty(last_vecs)

    # # 4개 eval length 각각에 대해 5개 샘플 정확도 읽고 mean/max 생성
    # row = {
    #     "video_id": vid,
    #     "npy_path": npy_path,
    #     "last_uncertainty": last_unc,
    # }
    # for el in (0, 1, 2, 3):
    #     accs = read_accuracies_all_samples(vid, el, S=S)  # shape (S,)
    #     row[f"acc_len{el}_mean"] = np.nanmean(accs) if np.isfinite(accs).any() else np.nan
    #     row[f"acc_len{el}_max"]  = np.nanmax(accs)  if np.isfinite(accs).any() else np.nan
    # records.append(row)
visualize_manifold_structure(np.array(all_samples_batch), intention_labels_batch)
# df = pd.DataFrame.from_records(records)
# df.to_csv(os.path.join(OUT_DIR, "uncertainty_vs_accuracy_summary.csv"), index=False)

# ---------- 5) 상관(피어슨/스피어만) ----------
def pearson_corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = (x[m] - x[m].mean()) / (x[m].std() + 1e-12)
    y = (y[m] - y[m].mean()) / (y[m].std() + 1e-12)
    return float((x*y).mean())

def spearman_corr(x, y):
    xr = pd.Series(x).rank(method="average").values
    yr = pd.Series(y).rank(method="average").values
    return pearson_corr(xr, yr)

corr_rows = []
targets = []
for el in (0,1,2,3):
    targets += [f"acc_len{el}_mean", f"acc_len{el}_max"]

for col in targets:
    r  = pearson_corr(df["last_uncertainty"].values, df[col].values)
    rh = spearman_corr(df["last_uncertainty"].values, df[col].values)
    corr_rows.append({"metric": col, "pearson_r": r, "spearman_rho": rh})

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(OUT_DIR, "uncertainty_accuracy_correlations.csv"), index=False)

# ---------- 6) 산점도 (mean / max 각각 4장씩 = 총 8장) ----------
for el in (0,1,2,3):
    for agg in ("mean", "max"):
        col = f"acc_len{el}_{agg}"
        x = df["last_uncertainty"].values
        y = df[col].values
        m = np.isfinite(x) & np.isfinite(y)

        plt.figure(figsize=(6,5))
        plt.scatter(x[m], y[m])
        plt.xlabel("Last-step temporal uncertainty (PR effective rank)")
        plt.ylabel(f"Accuracy @ eval_len={el} ({agg})")
        r  = pearson_corr(x, y)
        rh = spearman_corr(x, y)
        plt.title(f"Uncertainty vs Accuracy (len={el}, {agg})\nPearson r={r:.3f}, Spearman rho={rh:.3f}")
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"uncertainty_vs_accuracy_len{el}_{agg}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

print("Done:")
print(" - Summary CSV:", os.path.join(OUT_DIR, "uncertainty_vs_accuracy_summary.csv"))
print(" - Correlations CSV:", os.path.join(OUT_DIR, "uncertainty_accuracy_correlations.csv"))
print(" - Plots:", [os.path.join(OUT_DIR, f"uncertainty_vs_accuracy_len{el}_{agg}.png")
                    for el in (0,1,2,3) for agg in ("mean","max")])
