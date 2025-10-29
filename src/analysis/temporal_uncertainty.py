import os, re, glob
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 경로/패턴
# ------------------------------------------------------------
ROOT = "/home/hice1/skim3513/scratch/causdiff/outputs/infer_goal"
PATTERN = os.path.join(ROOT, "infer_goal_t0_*.npy")
OUT_DIR = os.path.join(ROOT, "processed_t0_png")
os.makedirs(OUT_DIR, exist_ok=True)

# infer_goal_t{step}_{YYYYmmdd_HHMMSS}.npy
_TS_RE = re.compile(r"infer_goal_t(?P<step>\d+)_(?P<ts>\d{8}_\d{6})\.npy$")

def _parse_ts(path: str):
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return int(m.group("step")), m.group("ts")

def _ts_key(ts: str):
    return ts.replace("_", "")  # YYYYmmddHHMMSS 기준 정렬

# ------------------------------------------------------------
# fallback temporal metrics (measure_uncertainty 없을 때 사용)
# ------------------------------------------------------------
def _stable_rank_from_cov(cov: np.ndarray, eps: float = 1e-12) -> float:
    tr = np.trace(cov)
    tr2 = np.sum(cov * cov)
    if tr2 <= eps: 
        return 0.0
    return float((tr * tr) / max(tr2, eps))

def _compute_temporal_metrics_fallback(samples: np.ndarray, window: int = 0):
    """
    samples: (S, T, D)
    return: dict[str, np.ndarray]
    """
    S, T, D = samples.shape
    m_t = samples.mean(axis=0)  # (T, D)

    stable_rank = np.zeros((T,), dtype=float)
    for t in range(T):
        X = samples[:, t, :]           # (S, D)
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(S - 1, 1)
        stable_rank[t] = _stable_rank_from_cov(cov)

    delta = np.zeros((T,), dtype=float)
    accel = np.zeros((T,), dtype=float)
    if T >= 2:
        delta[1:] = np.linalg.norm(np.diff(m_t, axis=0), axis=-1)
    if T >= 3:
        accel[2:] = np.linalg.norm(np.diff(m_t, n=2, axis=0), axis=-1)

    stable_rank_win = None
    if window and window > 1:
        stable_rank_win = np.zeros((T,), dtype=float)
        half = window // 2
        for t in range(T):
            t0 = max(0, t - half)
            t1 = min(T, t + half + 1)
            X = samples[:, t0:t1, :].reshape(-1, D)
            Xc = X - X.mean(axis=0, keepdims=True)
            cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
            stable_rank_win[t] = _stable_rank_from_cov(cov)

    return {
        "stable_rank_across_samples": stable_rank,
        "delta_l2_mean_feature": delta,
        "accel_l2_mean_feature": accel,
        "stable_rank_windowed": stable_rank_win
    }

def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + 1e-12)

# ------------------------------------------------------------
# 파일 수집/정렬
# ------------------------------------------------------------
pairs = []
for f in glob.glob(PATTERN):
    step, ts = _parse_ts(f)
    if step == 0 and ts is not None:
        pairs.append((f, ts))
if not pairs:
    raise FileNotFoundError("infer_goal_t0_* .npy 파일을 찾지 못했습니다.")

pairs.sort(key=lambda x: _ts_key(x[1]))
ordered = [p[0] for p in pairs]

print(f"[info] {len(ordered)} files for t0 found.")
print("       first:", os.path.basename(ordered[0]))
print("       last :", os.path.basename(ordered[-1]))

# ------------------------------------------------------------
# 순차 처리 루프 (PNG만 저장)
# ------------------------------------------------------------
for idx, path in enumerate(ordered):
    base = os.path.basename(path)
    _, ts = _parse_ts(path)

    arr = np.load(path, allow_pickle=False)  # 기대: (S, 1, T, D)
    samples = arr[:, 0, :, :]  # (S, T, D)

    # measure_uncertainty가 있으면 사용, 없으면 fallback
    if "measure_uncertainty" in globals() and callable(globals()["measure_uncertainty"]):
        try:
            res = measure_uncertainty(
                samples_STD=samples,
                class_embs=None,
                goal_emb=None,
                allowed_classes=None,
                tau=1.0,
                use_cosine=True,
                window=7
            )
            temporal = res.get("temporal", {})
        except Exception as e:
            print(f"[warn] measure_uncertainty 실패({e}). fallback 사용.")
            temporal = _compute_temporal_metrics_fallback(samples, window=7)
    else:
        temporal = _compute_temporal_metrics_fallback(samples, window=7)

    # 곡선 준비
    sr = temporal.get("stable_rank_across_samples", None)
    de = temporal.get("delta_l2_mean_feature", None)
    ac = temporal.get("accel_l2_mean_feature", None)

    # 최소한 하나라도 존재해야 그림을 그림
    series = []
    labels = []
    if sr is not None: series.append(_minmax(sr)); labels.append("stable_rank_across_samples (norm)")
    if de is not None: series.append(_minmax(de)); labels.append("delta_l2_mean_feature (norm)")
    if ac is not None: series.append(_minmax(ac)); labels.append("accel_l2_mean_feature (norm)")

    if not series:
        print(f"[skip] {base}: 그릴 수 있는 temporal 시계열이 없습니다.")
        continue

    # PNG 저장 (세 곡선을 한 장에)
    plt.figure(figsize=(10, 4.5))
    for y, lab in zip(series, labels):
        plt.plot(y, label=lab)
    plt.title(f"Temporal Uncertainty Waveform t0 @ {ts}")
    plt.xlabel("Time index (t)")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, f"temporal_waveform_t0_{ts}.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[{idx+1}/{len(ordered)}] saved PNG: {os.path.basename(png_path)}")

print("[done] 모든 t0 파일 PNG 저장 완료.")
