#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List

# -----------------------------
# 정렬 유틸
# -----------------------------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# -----------------------------
# 색상 팔레트 (클래스 고정)
# -----------------------------
def get_class_colors(C: int):
    import matplotlib
    base = [matplotlib.cm.get_cmap("tab20")(i) for i in range(20)]
    if C <= 20:
        return base[:C]
    # 부족하면 HSV 순환
    import colorsys
    extra = C - 20
    hsvs = [colorsys.hsv_to_rgb(i/extra, 0.75, 0.9) for i in range(extra)]
    hsvs = [(r, g, b, 1.0) for (r, g, b) in hsvs]
    return base + hsvs

# -----------------------------
# 연속 구간(RLE)
# -----------------------------
def rle_segments(labels_1d: np.ndarray):
    T = labels_1d.shape[0]
    if T == 0:
        return []
    segs = []
    start = 0
    cur = labels_1d[0]
    for t in range(1, T):
        if labels_1d[t] != cur:
            segs.append((start, t - start, cur))
            start = t
            cur = labels_1d[t]
    segs.append((start, T - start, cur))
    return segs

# -----------------------------
# 한 페어(B,P)를 한 그림에 그림 (하나의 Axes)
# -----------------------------
def plot_argmax_compare(b_arr: np.ndarray,
                        p_arr: np.ndarray,
                        out_path: str,
                        max_samples: int = 5,
                        class_names=None,
                        title: str = "Argmax Class Timelines (Baseline vs Proposed)"):
    """
    b_arr, p_arr: (S,T,C)
    한 그림 안에서 위쪽에 baseline S개, 아래쪽에 proposed S개를 이어서 배치
    """
    if b_arr.ndim == 4:
        b_arr = b_arr[:, 0, :, :]
        assert b_arr.ndim == 3
    if p_arr.ndim == 4:
        p_arr = p_arr[:, 0, :, :]
        assert p_arr.ndim == 3
    if b_arr.shape[1:] != p_arr.shape[1:]:
        raise ValueError(f"(T,C) mismatch: baseline{b_arr.shape} vs proposed{p_arr.shape}")
    S, T, C = b_arr.shape

    # argmax 레이블 (로짓이어도 동일)
    b_labels = np.argmax(b_arr, axis=-1)  # (S,T)
    p_labels = np.argmax(p_arr, axis=-1)  # (S,T)

    show_b = min(S, max_samples)
    show_p = min(p_arr.shape[0], max_samples)

    colors = get_class_colors(C)

    # 레이아웃: 위에 baseline show_b줄, 아래에 proposed show_p줄
    total_rows = show_b + show_p
    row_h = 0.8
    row_gap = 0.2
    h = max(2.0, 0.7 * total_rows + 1.2)

    fig = plt.figure(figsize=(12, h))
    ax = plt.gca()

    # baseline 영역
    for s in range(show_b):
        y = (total_rows - 1 - s) * (row_h + row_gap)  # 위에서부터 내려오도록
        for start, length, lab in rle_segments(b_labels[s]):
            ax.add_patch(Rectangle((start, y), length, row_h,
                                   facecolor=colors[lab], edgecolor='black', linewidth=0.3))
        ax.text(T + max(1, T * 0.01), y + row_h/2, f"Samp. {s+1} (B)", va="center", ha="left", fontsize=10)

    # 구분선
    sep_y = (total_rows - show_b) * (row_h + row_gap) - row_gap/2
    ax.hlines(sep_y, 0, T, colors='k', linestyles='dashed', linewidth=0.7)

    # proposed 영역
    for s in range(show_p):
        y = (show_p - 1 - s) * (row_h + row_gap)  # 아래쪽 블록
        for start, length, lab in rle_segments(p_labels[s]):
            ax.add_patch(Rectangle((start, y), length, row_h,
                                   facecolor=colors[lab], edgecolor='black', linewidth=0.3))
        ax.text(T + max(1, T * 0.01), y + row_h/2, f"Samp. {s+1} (P)", va="center", ha="left", fontsize=10)

    ax.set_xlim(0, T)
    ax.set_ylim(-row_gap, total_rows * (row_h + row_gap))
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_title(title)

    # 클래스 범례
    from matplotlib.lines import Line2D
    handles = []
    for c in range(C):
        name = f"{c}" if class_names is None else f"{c}: {class_names[c]}"
        handles.append(Line2D([0], [0], color=colors[c], lw=8, label=name))
    ncol = 6 if C > 24 else (4 if C > 16 else (3 if C > 12 else 2))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              ncol=ncol, frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# 파일 루프
# -----------------------------
def process_all(baseline_dir: str,
                proposed_dir: str,
                out_dir: str,
                pattern: str = "*.npy",
                max_samples: int = 5):
    b_files = list_sorted_files(baseline_dir, pattern)
    p_files = list_sorted_files(proposed_dir, pattern)
    if not b_files or not p_files:
        raise RuntimeError("No files found in one or both directories.")
    if len(b_files) != len(p_files):
        print(f"[WARN] count mismatch: baseline={len(b_files)}, proposed={len(p_files)}. Pairing by sorted index.")
    N = min(len(b_files), len(p_files))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {N} pairs...")
    for i in range(N):
        b_path, p_path = b_files[i], p_files[i]
        b_arr = np.load(b_path, allow_pickle=False)
        p_arr = np.load(p_path, allow_pickle=False)

        # 출력 폴더: baseline 파일명 stem 기준
        stem = os.path.splitext(os.path.basename(b_path))[0]
        pair_dir = os.path.join(out_dir, stem)
        os.makedirs(pair_dir, exist_ok=True)

        out_png = os.path.join(pair_dir, "argmax_timelines_compare.png")
        title = f"Argmax Timelines — {os.path.basename(b_path)}  vs  {os.path.basename(p_path)}"
        plot_argmax_compare(b_arr, p_arr, out_png, max_samples=max_samples, title=title)
        print(f"[{i:03d}] -> {out_png}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/baseline_30')
    ap.add_argument("--proposed_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/proposed_infer_action/30')
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--out_dir", type=str, default="/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/visualize_samples")
    ap.add_argument("--max_samples", type=int, default=5)
    args = ap.parse_args()

    process_all(args.baseline_dir, args.proposed_dir, args.out_dir, args.pattern, args.max_samples)

if __name__ == "__main__":
    main()
