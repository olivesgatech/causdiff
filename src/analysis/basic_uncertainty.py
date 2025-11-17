#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 정렬/확률/유틸
# -----------------------------
def _natural_key(s: str):
    # 'file12_a3.npy' -> ['file', 12, '_a', 3, '.npy']
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

def sanitize(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr

def looks_prob(arr: np.ndarray, atol: float = 1e-2) -> bool:
    """마지막 축(C)의 합이 1±atol 이고 음수 미존재 여부로 확률성 점검"""
    row_sum = np.nansum(arr, axis=-1)
    return np.all((row_sum > 1 - atol) & (row_sum < 1 + atol)) and np.nanmin(arr) > -atol

def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    ex_sum = np.nansum(ex, axis=-1, keepdims=True)
    ex_sum[ex_sum == 0] = 1.0
    return ex / ex_sum

def ensure_prob(arr: np.ndarray, force_softmax: bool = False) -> np.ndarray:
    """확률 텐서 보장. force_softmax=True면 무조건 softmax 적용"""
    if force_softmax:
        return softmax_last(arr)
    return arr if looks_prob(arr) else softmax_last(arr)


# -----------------------------
# Vote Entropy (S축 기준)
# -----------------------------
def vote_entropy_over_S(prob_stc: np.ndarray, log_base: float = np.e) -> np.ndarray:
    """
    prob_stc : (S, T, C) 확률 텐서
    반환      : (T,) 시간별 Vote Entropy

    알고리즘:
      - 각 t에 대해 labels_s = argmax_c prob_stc[s, t, c]
      - labels_s의 히스토그램 분포 p_t를 구함
      - H_t = - sum_k p_t[k] * log_base(p_t[k])
    """
    if prob_stc.ndim != 3:
        raise ValueError(f"Expected (S,T,C), got {prob_stc.shape}")
    S, T, C = prob_stc.shape

    labels_st = np.argmax(prob_stc, axis=-1)  # (S, T)
    H_t = np.zeros(T, dtype=np.float64)
    eps = 1e-12
    denom = np.log(log_base)

    for t in range(T):
        counts = np.bincount(labels_st[:, t], minlength=C).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            H_t[t] = 0.0
            continue
        p = counts / total
        p = np.clip(p, eps, 1.0)
        H_t[t] = -np.sum(p * (np.log(p) / denom))
    return H_t


# -----------------------------
# 플로팅
# -----------------------------
def plot_two_curves(h_base_t: np.ndarray, h_prop_t: np.ndarray, title: str, out_png: str, ylabel: str):
    fig = plt.figure(figsize=(10, 3.2))
    plt.plot(h_base_t, label="baseline", linewidth=2)
    plt.plot(h_prop_t, label="proposed", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# 처리 루틴 (단일 페어)
# -----------------------------
def process_pair(b_path: str,
                 p_path: str,
                 out_root: str,
                 force_softmax: bool = False,
                 log_base: float = np.e) -> dict:
    b = np.load(b_path, allow_pickle=False)
    p = np.load(p_path, allow_pickle=False)
    if b.ndim == 4:
        b = b[:, 0, :, :]
        assert b.ndim == 3
    if p.ndim == 4:
        p = p[:, 0, :, :]
        assert p.ndim == 3

    if b.shape[1:] != p.shape[1:]:
        raise ValueError(f"(T,C) mismatch: {os.path.basename(b_path)} {b.shape} vs {os.path.basename(p_path)} {p.shape}")
    S, T, C = b.shape

    #b = sanitize(b); p = sanitize(p)
    bprob = b#ensure_prob(b, force_softmax=force_softmax)
    pprob = p#ensure_prob(p, force_softmax=force_softmax)

    # Vote Entropy 계산
    b_H_t = vote_entropy_over_S(bprob, log_base=log_base)
    p_H_t = vote_entropy_over_S(pprob, log_base=log_base)
    diff_t = p_H_t - b_H_t

    # 출력 경로
    stem_b = os.path.splitext(os.path.basename(b_path))[0]
    pair_dir = os.path.join(out_root, stem_b)
    os.makedirs(pair_dir, exist_ok=True)

    unit = "nat" if np.isclose(log_base, np.e) else ("bit" if np.isclose(log_base, 2.0) else f"log{log_base}")
    # 저장
    np.save(os.path.join(pair_dir, f"baseline_vote_entropy_{unit}.npy"), b_H_t)
    np.save(os.path.join(pair_dir, f"proposed_vote_entropy_{unit}.npy"), p_H_t)
    np.save(os.path.join(pair_dir, f"delta_vote_entropy_{unit}.npy"), diff_t)

    np.savetxt(os.path.join(pair_dir, f"baseline_vote_entropy_{unit}.csv"), b_H_t, delimiter=",", fmt="%.8f")
    np.savetxt(os.path.join(pair_dir, f"proposed_vote_entropy_{unit}.csv"), p_H_t, delimiter=",", fmt="%.8f")
    np.savetxt(os.path.join(pair_dir, f"delta_vote_entropy_{unit}.csv"), diff_t, delimiter=",", fmt="%.8f")

    # 플롯
    plot_two_curves(b_H_t, p_H_t,
                    title=f"Vote Entropy over Time ({unit})",
                    out_png=os.path.join(pair_dir, f"vote_entropy_compare_{unit}.png"),
                    ylabel=f"Vote Entropy ({unit})")

    return {
        "pair_dir": pair_dir,
        "baseline": os.path.basename(b_path),
        "proposed": os.path.basename(p_path),
        "unit": unit
    }


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str,  default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3/baselines_20_l3', help="baseline .npy가 있는 폴더")
    ap.add_argument("--proposed_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3/proposed_20', help="proposed .npy가 있는 폴더")
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--out_dir", type=str, default="/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3/outputs_basic_uncertainty_20")
    ap.add_argument("--force_softmax", action="store_true", help="입력이 로짓 등일 때 C축으로 softmax 강제 적용")
    ap.add_argument("--log_base", type=float, default=np.e, help="로그 밑 (기본 e). 2로 주면 비트 단위.")
    args = ap.parse_args()

    b_files = list_sorted_files(args.baseline_dir, args.pattern)
    p_files = list_sorted_files(args.proposed_dir, args.pattern)
    if len(b_files) == 0 or len(p_files) == 0:
        raise RuntimeError("No files found in one or both directories.")
    if len(b_files) != len(p_files):
        print(f"[WARN] count mismatch: baseline={len(b_files)}, proposed={len(p_files)}. Pairing by sorted index.")

    N = min(len(b_files), len(p_files))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Processing {N} pairs...")
    for i in range(N):
        info = process_pair(
            b_files[i], p_files[i], args.out_dir,
            force_softmax=args.force_softmax,
            log_base=args.log_base
        )
        print(f"[{i:03d}] {os.path.basename(b_files[i])}  ||  {os.path.basename(p_files[i])}  ->  {info['pair_dir']} ({info['unit']})")


if __name__ == "__main__":
    main()
