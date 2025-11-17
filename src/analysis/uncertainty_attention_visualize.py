import os, glob, re, json
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 자연 정렬 & 경로 유틸
# -----------------------------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# -----------------------------
# 수치/확률/엔트로피 유틸
# -----------------------------
def sanitize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr

def looks_prob(arr: np.ndarray, tol=1e-3) -> bool:
    if arr.ndim < 1: return False
    row_sum = np.nansum(arr, axis=-1)
    return np.all((row_sum > 1 - tol) & (row_sum < 1 + tol)) and np.nanmin(arr) >= -tol

def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    s = np.nansum(ex, axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return ex / s

def ensure_prob(arr: np.ndarray) -> np.ndarray:
    return arr if looks_prob(arr) else softmax_last(arr)

def entropy_over_last(p: np.ndarray, eps=1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

def minmax_normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    y = (x - xmin) / (xmax - xmin)
    y[~np.isfinite(y)] = 0.0
    return y

# -----------------------------
# 중앙 3/8 구간 슬라이스 인덱스
# -----------------------------
def center_slice_indices(T: int, frac: float = 3.0/8.0) -> Tuple[int, int]:
    L = max(1, int(round(T * frac)))
    start = max(0, (T - L) // 2)
    end = min(T, start + L)
    return start, end

# -----------------------------
# 어텐션 로딩 (S,T) 형태 유지 (추가 정규화 없음)
# -----------------------------
def load_attention_ST(attn_path: str) -> np.ndarray:
    """
    반환: (S, T)
      - (S, 1, T)  → (S, T)
      - (S, T)     → 그대로
      - (T,)       → (1, T)
    """
    a = np.load(attn_path, allow_pickle=False)
    a = sanitize(a)

    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]           # (S, T)
    elif a.ndim == 2:
        pass                     # (S, T)
    elif a.ndim == 1:
        a = a[None, :]           # (1, T)
    else:
        raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")
    return a  # (S, T)

# -----------------------------
# 불확실성: S축 통합 → (T,)
# -----------------------------
def time_unc_from_samples(stc_path: str) -> Tuple[np.ndarray, dict]:
    """
    입력: (S,T,C) 또는 (S,B,T,C) — B가 있으면 B=0만 사용.
    처리:
      1) 확률화
      2) S축 평균 → (T,C)
      3) 엔트로피 → (T,)
    """
    x = np.load(stc_path, allow_pickle=False)
    if x.ndim == 4:
        x = x[:, 0, :, :]   # (S,T,C)
    elif x.ndim != 3:
        raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C) or (S,B,T,C), got {x.shape}")
    S, T, C = x.shape
    probs = ensure_prob(sanitize(x))      # (S,T,C)
    p_bar = probs.mean(axis=0)            # (T,C)
    H_t = entropy_over_last(p_bar)        # (T,)
    return H_t, {"S": S, "T": T, "C": C}

# -----------------------------
# 상관/플로팅
# -----------------------------
def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def plot_overlay_entropy_and_attn(Hn_t: np.ndarray, attn_ST: np.ndarray, title: str, out_path: str):
    """
    Hn_t: (T_seg,)      - 정규화된 엔트로피(슬라이스)
    attn_ST: (S,T_seg)  - 샘플별 어텐션(슬라이스, 재정규화 없음)
    """
    T_use = min(Hn_t.size, attn_ST.shape[1])
    x = np.arange(T_use)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.0), sharex=True)

    # 1행: 정규화된 불확실성 (단일 곡선)
    ax1 = axes[0]
    ax1.plot(x, Hn_t[:T_use], linewidth=2.0)
    ax1.set_ylabel("Normalized Entropy (segment)")
    ax1.set_title(title + " — Uncertainty (center 3/8)")
    ax1.grid(True, alpha=0.3)

    # 2행: 어텐션 (S개 오버레이, 원래 스케일)
    ax2 = axes[1]
    S = attn_ST.shape[0]
    for s in range(S):
        ax2.plot(x, attn_ST[s, :T_use], linewidth=1.2, alpha=0.9, label=f"S{s}")
    ax2.set_xlabel("Time (segment)")
    ax2.set_ylabel("Attention (original scale)")
    ax2.set_title("Attention Overlay (center 3/8)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", ncol=min(S, 5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------------
# 실행 루틴
# -----------------------------
def run_with_shared_attention(
    infergoal_dir: str,
    attn_dir: str,
    out_root: str,
    pattern: str = "*.npy"
):
    os.makedirs(out_root, exist_ok=True)
    g_files = list_sorted_files(infergoal_dir, pattern)
    a_files = list_sorted_files(attn_dir, pattern)

    if not g_files or not a_files:
        raise RuntimeError("One or more folders are empty.")

    N = min(len(g_files), len(a_files))
    if len({len(g_files), len(a_files)}) != 1:
        print(f"[WARN] count mismatch: infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

    out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

    print(f"Using shared attention from: {attn_dir}")
    print(f"#pairs (by index) = {N}")

    # 집계 버킷
    per_file_max_r = []   # 각 파일에서 max_s r
    all_sample_r = []     # 모든 파일×샘플 r
    rows = []             # CSV 요약

    for i in range(N):
        g_path = g_files[i]
        a_path = a_files[i]
        stem   = os.path.splitext(os.path.basename(g_path))[0]

        # (S,T) 어텐션 로드
        attn_ST_full = load_attention_ST(a_path)  # (S_a, T_a)

        # 불확실성: S축 통합 → (T,)
        H_full, meta = time_unc_from_samples(g_path)  # (T,)

        # 공통 길이 및 중앙 3/8 구간 인덱스
        
        T_common = min(H_full.size, attn_ST_full.shape[1])
        start = 0#int(T_common * 2 / 8)
        #end = int(T_common * 4 / 8)#T_common
        #end = int(T_common * 5 / 8)
        #end = int(T_common * 6 / 8)
        end = int(T_common * 8 / 8)

        # 슬라이스 (어텐션 재정규화 없음)
        H_seg = H_full[start:end]                  # (T_seg,)
        attn_seg = attn_ST_full[:, start:end]   # (S, T_seg)

        # 엔트로피만 구간 기준 min-max 정규화
        Hn_seg = H_seg
        #Hn_seg = minmax_normalize_1d(H_seg)                   # (T_seg,)

        # 플롯
        out_png = os.path.join(out_g, f"{stem}_overlay_center3of8.png")
        plot_overlay_entropy_and_attn(
            Hn_t=Hn_seg,
            attn_ST=attn_seg,
            title=f"infer_goal: {stem}",
            out_path=out_png
        )
        print(f"[{i:03d}] saved overlay (center 3/8) -> {out_png}")

        # 샘플별 상관계수: corr(attn[s], Hn_seg)  (어텐션 재정규화 없이 원 스케일 사용)
        r_list = []
        for s in range(attn_seg.shape[0]):
            r = pearson_corr(attn_seg[s], Hn_seg)
            r_list.append(r)
            rows.append({
                "idx": i,
                "sample": s,
                "infer_goal_file": os.path.basename(g_path),
                "attention_file": os.path.basename(a_path),
                "r_corr": r,
                "S": meta["S"], "T": meta["T"], "C": meta["C"],
                "T_common": T_common,
                "slice_start": int(start),
                "slice_end": int(end),
            })

        if r_list:
            per_file_max_r.append(float(np.min(r_list)))
            all_sample_r.extend(r_list)

    # 통계 요약
    max_r_mean = float(np.mean(per_file_max_r)) if per_file_max_r else 0.0
    overall_r_mean = float(np.mean(all_sample_r)) if all_sample_r else 0.0

    # CSV/JSON 저장
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "summary_shared_attention_by_index.csv")
    df.to_csv(csv_path, index=False)

    agg = {
        "num_files": int(len(per_file_max_r)),
        "num_rows": int(len(rows)),
        "mean_of_max_r_per_file": max_r_mean,   # 각 파일의 max 상관계수들의 평균
        "overall_mean_r": overall_r_mean,       # 모든 샘플 상관계수의 단순 평균
        "analyzed_segment": "center 3/8 of timeline (attention not renormalized)",
        "summary_csv": csv_path,
        "out_root": out_root
    }
    with open(os.path.join(out_root, "aggregate_shared_attention_by_index.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== Aggregate (shared attention; CENTER 3/8, no attn renorm) ===")
    print(json.dumps(agg, indent=2))

def plot_top5_negative_pairs(pairs, out_path: str, normalize_entropy: bool = True):
    """
    pairs: list of dicts with keys:
        'stem', 'sample', 'r', 'H_seg' (np.ndarray, shape (T_seg,)),
        'A_seg' (np.ndarray, shape (T_seg,))
    out_path: save path for the single figure
    normalize_entropy: 시각화에서 엔트로피를 [0,1]로 민맥스 정규화할지 여부(분석 결과엔 영향 없음)
    """
    import matplotlib.pyplot as plt
    # r 오름차순(가장 음수 5개)
    pairs_sorted = sorted(pairs, key=lambda d: d['r'])
    top5 = pairs_sorted[:min(5, len(pairs_sorted))]
    if not top5:
        print("[WARN] No pairs to plot for top-5 negative overlay.")
        return

    rows = len(top5)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.8*rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, item in zip(axes, top5):
        H = item['H_seg']
        A = item['A_seg']
        T_seg = min(H.size, A.size)
        x = np.arange(T_seg)

        # 좌축: 엔트로피
        yH = H
        if normalize_entropy:
            yH = minmax_normalize_1d(H)

        ax.plot(x, yH[:T_seg], linewidth=2.0, label="Entropy (norm)" if normalize_entropy else "Entropy")
        ax.set_ylabel("Entropy" + (" (norm)" if normalize_entropy else ""))
        ax.grid(True, alpha=0.3)

        # 우축: 어텐션 (원 스케일)
        ax2 = ax.twinx()
        ax2.plot(x, A[:T_seg], linewidth=1.2, alpha=0.9, color="C1", label="Attention (orig)")
        ax2.set_ylabel("Attention (orig)")

        title = f"{item['stem']} | s={item['sample']} | r={item['r']:.3f}"
        ax.set_title(title)

    axes[-1].set_xlabel("Time (selected segment)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[TOP-5] saved -> {out_path}")

def plot_top5_mean_std(pairs, out_path: str, normalize_entropy: bool = True):
    """
    pairs: list of dicts with keys:
        'stem', 'sample', 'r', 'H_seg' (np.ndarray), 'A_seg' (np.ndarray)
    1) r 오름차순으로 가장 음수 5개 선정
    2) 공통 최소 길이로 트림
    3) (엔트로피는 옵션으로 min-max 정규화) 후 mean±std 계산
    4) 단일 플롯에 twin-y로 겹쳐 그림
       - 엔트로피: 진분홍 실선, 연분홍 ±1SD
       - 어텐션: 파란 실선, 연파랑 ±1SD
    """
    if not pairs:
        print("[WARN] No pairs to aggregate.")
        return

    # 1) top-5 by most negative r
    pairs_sorted = sorted(pairs, key=lambda d: d['r'])
    top5 = pairs_sorted[:min(4, len(pairs_sorted))]
    if len(top5) == 0:
        print("[WARN] No pairs selected for top-5.")
        return

    # 2) align by common minimal length
    T_min = min(min(len(p['H_seg']), len(p['A_seg'])) for p in top5)

    H_stack, A_stack = [], []
    for p in top5:
        H = p['H_seg'][:T_min]
        A = p['A_seg'][:T_min]
        if normalize_entropy:
            H = minmax_normalize_1d(H)
        H_stack.append(H)
        A_stack.append(A)
    H_stack = np.stack(H_stack, axis=0)  # (K<=5, T_min)
    A_stack = np.stack(A_stack, axis=0)  # (K<=5, T_min)

    # 3) mean ± std
    if H_stack.shape[0] > 1:
        H_mu = H_stack.mean(axis=0); H_sd = H_stack.std(axis=0, ddof=1)
    else:
        H_mu = H_stack[0]; H_sd = np.zeros_like(H_mu)

    if A_stack.shape[0] > 1:
        A_mu = A_stack.mean(axis=0); A_sd = A_stack.std(axis=0, ddof=1)
    else:
        A_mu = A_stack[0]; A_sd = np.zeros_like(A_mu)

    x = np.arange(T_min)

    # 4) single plot with twin y-axes
    fig, ax1 = plt.subplots(figsize=(5, 2))

    # Entropy on left axis (hot pink)
    ent_line = ax1.plot(x, H_mu, linewidth=2.0, color="#FF1493", label="Uncertainty")[0]  # 진분홍
    ax1.fill_between(x, H_mu - H_sd, H_mu + H_sd, alpha=0.25, color="#FF69B4", label="Entropy ±1 SD")  # 연분홍
    ax1.set_ylabel("Entropy" + (" (norm)" if normalize_entropy else ""))

    # Attention on right axis (blue)
    ax2 = ax1.twinx()
    att_line = ax2.plot(x, A_mu, linewidth=2.0, color="#1f77b4", label="Attention")[0]  # 파랑
    ax2.fill_between(x, A_mu - A_sd, A_mu + A_sd, alpha=0.25, color="#1f77b4", label="Attention ±1 SD")
    ax2.set_ylabel("Attention (original scale)")

    ax1.set_xlabel("Time (selected segment)")
    ax1.set_title("Top-5 most negative pairs — mean ± std (single plot)")
    ax1.grid(True, alpha=0.3)

    # 공동 범례 구성
    lines = [ent_line, att_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[TOP-5 mean±std singleplot] saved -> {out_path}")


def run_with_shared_attention(
    infergoal_dir: str,
    attn_dir: str,
    out_root: str,
    pattern: str = "*.npy"
):
    os.makedirs(out_root, exist_ok=True)
    g_files = list_sorted_files(infergoal_dir, pattern)
    a_files = list_sorted_files(attn_dir, pattern)

    if not g_files or not a_files:
        raise RuntimeError("One or more folders are empty.")

    N = min(len(g_files), len(a_files))
    if len({len(g_files), len(a_files)}) != 1:
        print(f"[WARN] count mismatch: infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

    out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

    print(f"Using shared attention from: {attn_dir}")
    print(f"#pairs (by index) = {N}")

    per_file_max_r = []
    all_sample_r = []
    rows = []

    # ★ 여기 리스트에 각 샘플-쌍의 시계열을 저장해 나중에 top5를 그립니다.
    all_pairs_series = []  # dicts with keys: stem, sample, r, H_seg, A_seg

    for i in range(N):
        g_path = g_files[i]
        a_path = a_files[i]
        stem   = os.path.splitext(os.path.basename(g_path))[0]

        attn_ST_full = load_attention_ST(a_path)     # (S_a, T_a)
        H_full, meta = time_unc_from_samples(g_path) # (T,)

        # --- 사용 구간 설정(현재 코드의 start/end를 그대로 따름) ---
        T_common = min(H_full.size, attn_ST_full.shape[1])
        start = 0
        end = int(T_common * 8 / 8)
        # ---------------------------------------------------------

        H_seg = H_full[start:end]                    # (T_seg,)
        attn_seg = attn_ST_full[:, start:end]       # (S, T_seg)

        # 개별 샘플 r 및 시계열 저장
        r_list = []
        for s in range(attn_seg.shape[0]):
            A = attn_seg[s]
            r = pearson_corr(A, H_seg)
            r_list.append(r)
            rows.append({
                "idx": i, "sample": s,
                "infer_goal_file": os.path.basename(g_path),
                "attention_file": os.path.basename(a_path),
                "r_corr": r,
                "S": meta["S"], "T": meta["T"], "C": meta["C"],
                "T_common": T_common,
                "slice_start": int(start),
                "slice_end": int(end),
            })
            # ★ top5 선정을 위한 시계열 보관
            all_pairs_series.append({
                "stem": stem,
                "sample": s,
                "r": float(r),
                "H_seg": H_seg.copy(),
                "A_seg": A.copy(),
            })

        if r_list:
            per_file_max_r.append(float(np.min(r_list)))  # "가장 음의"를 파일 대표로 쓸 거면 min 사용
            all_sample_r.extend(r_list)

        # 개별 파일 오버레이(원하면 유지)
        out_png = os.path.join(out_g, f"{stem}_overlay_center3of8.png")
        plot_overlay_entropy_and_attn(
            Hn_t=H_seg,            # 시각화에서 엔트로피 정규화 원하면 minmax_normalize_1d(H_seg)로 바꿔도 됨
            attn_ST=attn_seg,
            title=f"infer_goal: {stem}",
            out_path=out_png
        )
        print(f"[{i:03d}] saved overlay -> {out_png}")

    # 통계 요약
    max_r_mean = float(np.mean(per_file_max_r)) if per_file_max_r else 0.0
    overall_r_mean = float(np.mean(all_sample_r)) if all_sample_r else 0.0

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "summary_shared_attention_by_index.csv")
    df.to_csv(csv_path, index=False)

    agg = {
        "num_files": int(len(per_file_max_r)),
        "num_rows": int(len(rows)),
        "mean_of_max_r_per_file": max_r_mean,
        "overall_mean_r": overall_r_mean,
        "analyzed_segment": "custom slice (see slice_start/slice_end in CSV)",
        "summary_csv": csv_path,
        "out_root": out_root
    }
    with open(os.path.join(out_root, "aggregate_shared_attention_by_index.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n=== Aggregate (shared attention; custom slice) ===")
    print(json.dumps(agg, indent=2))

    # ★ 가장 강한 음의 상관관계 5개 쌍을 한 Figure에 오버레이
    top5_path = os.path.join(out_root, "top5_negative_pairs_overlay.png")
    plot_top5_negative_pairs(all_pairs_series, top5_path, normalize_entropy=True)

    top5_avg_path = os.path.join(out_root, "top5_negative_pairs_mean_std.png")
    plot_top5_mean_std(all_pairs_series, top5_avg_path, normalize_entropy=True)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    #BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects"
    #BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3"
    BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2"
    INFERGOAL_DIR = os.path.join(BASE, "infer_goal/20")
    ATTENTION_DIR = os.path.join(BASE, "attention_map_20")
    OUT = os.path.join(BASE, "uncertainty_attention_out_shared_20")

    run_with_shared_attention(
        infergoal_dir=INFERGOAL_DIR,
        attn_dir=ATTENTION_DIR,
        out_root=OUT,
        pattern="*.npy"
    )
