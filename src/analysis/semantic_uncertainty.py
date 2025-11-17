import os
import glob
import math
import argparse
from typing import Dict, List, Tuple
import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Semantic groups (중복 허용)
# -----------------------------
SEMANTIC_GROUPS = {
    "pancake":           [0,2,3,4,5,6,7,8,14,24,25,27,28,31,32,33,39,40,41,45],
    "coffee":            [1,5,6,15,18,19,20,21,22,23,25,27,28,30,34,38,40,42,44],
    "kitchen_cleaning":  [9,10,11,12,34,40],
    "device":            [13,34,36,37],
    "dining":            [15,17,27,28,31,32,33,34,40],
    "dish_cleaning":     [16,26,34,35,40,43,46],
    # 지정되지 않은 클래스는 'other'로
}

# -----------------------------
# 2) 유틸 함수
# -----------------------------
def _natural_key(s: str):
    # 'file12_a3.npy' -> ['file', 12, '_a', 3, '.npy']
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
#     return sorted([p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)])

def sanitize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr

def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    ex_sum = np.nansum(ex, axis=-1, keepdims=True)
    ex_sum[ex_sum == 0] = 1.0
    return ex / ex_sum

def looks_prob(arr: np.ndarray) -> bool:
    row_sum = np.nansum(arr, axis=-1)
    return np.all((row_sum > 1-1e-2) & (row_sum < 1+1e-2)) and np.nanmin(arr) > -1e-4

def ensure_prob(arr: np.ndarray) -> np.ndarray:
    return arr #if looks_prob(arr) else softmax_last(arr)

def build_class_to_groups(num_classes: int, group_dict: Dict[str, List[int]]):
    group_names = list(group_dict.keys())
    explicit = {k: set(v) for k, v in group_dict.items()}
    assigned = set().union(*explicit.values()) if explicit else set()
    need_other = len(assigned) < num_classes
    if need_other:
        group_names.append("other")
    name2idx = {n:i for i,n in enumerate(group_names)}

    class_to_groups = [[] for _ in range(num_classes)]
    for name, cls_set in explicit.items():
        gi = name2idx[name]
        for c in cls_set:
            if 0 <= c < num_classes:
                class_to_groups[c].append(gi)

    if need_other:
        other_idx = name2idx["other"]
        for c in range(num_classes):
            if not class_to_groups[c]:
                class_to_groups[c].append(other_idx)

    per_class_norm = [1.0/len(g) if len(g)>0 else 0.0 for g in class_to_groups]
    return class_to_groups, per_class_norm, group_names

def classes_to_group_probs(prob_stc: np.ndarray,
                           class_to_groups: List[List[int]],
                           per_class_norm: List[float],
                           num_groups: int) -> np.ndarray:
    S,T,C = prob_stc.shape
    G = num_groups
    out = np.zeros((S,T,G), dtype=np.float64)
    
    for c in range(C):
        if not class_to_groups[c]:
            continue
        share = per_class_norm[c]
        for g in class_to_groups[c]:
            out[..., g] += prob_stc[..., c]# * share
    row_sum = out.sum(axis=-1, keepdims=True)
    row_sum[row_sum==0] = 1.0
    return out / row_sum

def entropy_over_groups(p_stg: np.ndarray, eps=1e-12) -> np.ndarray:
    p = np.clip(p_stg, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)  # (S,T)

# -----------------------------
# 3) 플롯 스타일(고정 팔레트)
# -----------------------------
def get_group_colors(group_names: List[str]):
    # 그룹 수가 적정(<=10)이라 가정하고, 충분히 구분되는 기본 컬러셋 사용
    base_colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    if len(group_names) > len(base_colors):
        # 부족하면 HSV 순환
        import colorsys
        extra = len(group_names) - len(base_colors)
        hsvs = [colorsys.hsv_to_rgb(i/extra, 0.65, 0.9) for i in range(extra)]
        to_hex = lambda rgb: "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
        base_colors += [to_hex(c) for c in hsvs]
    return base_colors[:len(group_names)]

def save_group_legend(group_names: List[str], colors: List[str], out_path: str):
    fig = plt.figure(figsize=(max(4, len(group_names)*0.6), 1.2))
    for i,(name,col) in enumerate(zip(group_names, colors)):
        plt.plot([],[], marker="s", linestyle="None", markersize=10, label=f"{i}: {name}", color=col)
    plt.legend(ncol=min(5,len(group_names)), frameon=False, loc="center")
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------------
# 4) 시각화
# -----------------------------
def plot_group_heatmap(argmax_st: np.ndarray, colors: List[str], title: str, out_path: str):
    """
    argmax_st: (S,T) 정수(그룹 id)
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    G = len(colors)
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, G+0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(12, max(3, argmax_st.shape[0]*0.25)))
    plt.imshow(argmax_st, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    plt.colorbar(ticks=list(range(G)), fraction=0.02)
    plt.xlabel("Time")
    plt.ylabel("Sample")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_diff_mask(argmax_base: np.ndarray, argmax_prop: np.ndarray, title: str, out_path: str):
    diff = (argmax_base != argmax_prop).astype(np.float64)
    fig = plt.figure(figsize=(12, max(3, diff.shape[0]*0.25)))
    plt.imshow(diff, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.xlabel("Time")
    plt.ylabel("Sample")
    plt.title(title + " (white=changed group)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_time_uncertainty(h_base_t: np.ndarray, h_prop_t: np.ndarray, out_path: str):
    fig = plt.figure(figsize=(12, 3.2))
    plt.plot(h_base_t, label="baseline", linewidth=2)
    plt.plot(h_prop_t, label="proposed", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Entropy (mean over S)")
    plt.title("Time-wise semantic uncertainty")
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_stack_area(mean_stg_tg: np.ndarray, group_names: List[str], colors: List[str], title: str, out_path: str):
    """
    mean_stg_tg: (T,G)  시간별 그룹 평균 확률
    """
    T,G = mean_stg_tg.shape
    x = np.arange(T)
    ys = [mean_stg_tg[:,g] for g in range(G)]
    fig = plt.figure(figsize=(12, 3.2))
    plt.stackplot(x, ys, labels=[f"{i}:{n}" for i,n in enumerate(group_names)], colors=colors)
    plt.xlabel("Time")
    plt.ylabel("Mean group prob (over S)")
    plt.title(title)
    plt.legend(loc="upper right", ncol=min(5,len(group_names)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -----------------------------
# 5) 파일 처리 파이프라인
# -----------------------------
def process_pair(b_path: str, p_path: str, out_dir: str):
    b = np.load(b_path, allow_pickle=False)
    p = np.load(p_path, allow_pickle=False)
    if b.ndim != 3 or p.ndim != 3:
        raise ValueError("Both arrays must be (S,T,C).")

    #b = sanitize(b); p = sanitize(p)
    if b.shape[1:] != p.shape[1:]:
        raise ValueError(f"(T,C) mismatch: {os.path.basename(b_path)} {b.shape} vs {os.path.basename(p_path)} {p.shape}")
    S,T,C = b.shape

    # 확률화
    bprob = ensure_prob(b)
    pprob = ensure_prob(p)

    # 그룹 매핑
    class_to_groups, per_class_norm, group_names = build_class_to_groups(C, SEMANTIC_GROUPS)
    #print(class_to_groups, per_class_norm, group_names)
    G = len(group_names)
    colors = get_group_colors(group_names)

    # (S,T,G)
    b_stg = classes_to_group_probs(bprob, class_to_groups, per_class_norm, G)
    p_stg = classes_to_group_probs(pprob, class_to_groups, per_class_norm, G)

    # 히트맵용 argmax (S,T)
    b_arg = np.argmax(b_stg, axis=-1)
    p_arg = np.argmax(p_stg, axis=-1)

    # 시간별 불확실성 (S 평균)
    b_H_t = entropy_over_groups(b_stg).mean(axis=0)
    p_H_t = entropy_over_groups(p_stg).mean(axis=0)

    # 시간별 그룹 평균 확률 (S 평균) → (T,G)
    b_mean_tg = b_stg.mean(axis=0)
    p_mean_tg = p_stg.mean(axis=0)

    # 출력 경로들
    stem = os.path.splitext(os.path.basename(b_path))[0]
    pair_dir = os.path.join(out_dir, f"{stem}")
    os.makedirs(pair_dir, exist_ok=True)

    # 범례
    save_group_legend(group_names, colors, os.path.join(pair_dir, "legend.png"))

    # 히트맵
    plot_group_heatmap(b_arg, colors, f"Baseline groups: {os.path.basename(b_path)}", os.path.join(pair_dir, "baseline_groups.png"))
    plot_group_heatmap(p_arg, colors, f"Proposed groups: {os.path.basename(p_path)}", os.path.join(pair_dir, "proposed_groups.png"))

    # 차이 마스크
    plot_diff_mask(b_arg, p_arg, "Changed group positions", os.path.join(pair_dir, "group_change_mask.png"))

    # 시간별 불확실성
    plot_time_uncertainty(b_H_t, p_H_t, os.path.join(pair_dir, "time_uncertainty.png"))

    # 스택 영역도 (베이스라인/프로포즈드 각각)
    plot_stack_area(b_mean_tg, group_names, colors, "Baseline: time-wise group composition (mean over S)", os.path.join(pair_dir, "baseline_group_stack.png"))
    plot_stack_area(p_mean_tg, group_names, colors, "Proposed: time-wise group composition (mean over S)", os.path.join(pair_dir, "proposed_group_stack.png"))

    # 간단 로그
    return {
        "pair_dir": pair_dir,
        "legend": "legend.png",
        "baseline_groups": "baseline_groups.png",
        "proposed_groups": "proposed_groups.png",
        "group_change_mask": "group_change_mask.png",
        "time_uncertainty": "time_uncertainty.png",
        "baseline_group_stack": "baseline_group_stack.png",
        "proposed_group_stack": "proposed_group_stack.png",
        "group_names": group_names
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/baselines_20')
    ap.add_argument("--proposed_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/proposed_20')
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--out_dir", type=str, default="./outputs/semantic_uncertainty_out")
    args = ap.parse_args()

    b_files = list_sorted_files(args.baseline_dir, args.pattern)
    p_files = list_sorted_files(args.proposed_dir, args.pattern)
    print(b_files)
    print(p_files)
    if len(b_files) == 0 or len(p_files) == 0:
        raise RuntimeError("No files found in one or both directories.")
    if len(b_files) != len(p_files):
        print(f"[WARN] count mismatch: baseline={len(b_files)}, proposed={len(p_files)}. Pairing by sorted index.")
    N = min(len(b_files), len(p_files))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Processing {N} pairs...")
    for i in range(N):
        info = process_pair(b_files[i], p_files[i], args.out_dir)
        print(f"[{i:03d}] -> {info['pair_dir']}")

if __name__ == "__main__":
    main()
