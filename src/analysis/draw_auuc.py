#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, csv
from typing import List, Tuple
import numpy as np

# -----------------------------
# 정렬/유틸
# -----------------------------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str="*.npy") -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# -----------------------------
# Expected Entropy
# -----------------------------
def expected_entropy_over_S(prob_stc: np.ndarray, log_base: float=np.e, eps: float=1e-12) -> np.ndarray:
    # prob_stc: (S,T,C) -> (T,)
    if prob_stc.ndim != 3:
        raise ValueError(f"Expected (S,T,C), got {prob_stc.shape}")
    p = np.clip(prob_stc, eps, 1.0)
    H_st = -np.sum(p * (np.log(p) / np.log(log_base)), axis=-1)  # (S,T)
    return H_st.mean(axis=0)  # (T,)

def compute_curve(npy_path: str, force_softmax: bool, log_base: float) -> np.ndarray:
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim == 4:  # (S,B,T,C) -> B=0
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{npy_path}: expected (S,T,C) or (S,1,T,C), got {arr.shape}")
    prob = arr #ensure_prob(arr, force_softmax)
    return expected_entropy_over_S(prob, log_base)  # (T,)

def get_T_len(npy_path: str) -> int:
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{npy_path}: expected (S,T,C), got {arr.shape}")
    return arr.shape[1]

# -----------------------------
# 그룹 평균 곡선 구성
# -----------------------------
def build_group_mean(proposed_or_baseline_files: List[str],
                     force_softmax: bool,
                     log_base: float,
                     T_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    return:
      mean_t: (T_max,) 시간별 평균 (NaN 제외)
      cnt_t : (T_max,) 각 t에서 유효 파일 수
    """
    if len(proposed_or_baseline_files) == 0:
        return np.full(T_max, np.nan), np.zeros(T_max, dtype=int)

    mat = np.full((len(proposed_or_baseline_files), T_max), np.nan, dtype=np.float64)
    for i, fp in enumerate(proposed_or_baseline_files):
        curve = compute_curve(fp, force_softmax, log_base)  # (T_i,)
        Ti = min(len(curve), T_max)
        if Ti > 0:
            mat[i, :Ti] = curve[:Ti]

    mean_t = np.nanmean(mat, axis=0)
    cnt_t  = np.sum(~np.isnan(mat), axis=0).astype(int)
    return mean_t, cnt_t

# -----------------------------
# 구간 적분 (양쪽 유효 교집합만)
# -----------------------------
def integrate_over_range(b_mean: np.ndarray, p_mean: np.ndarray,
                         b_cnt: np.ndarray, p_cnt: np.ndarray,
                         start_idx: int, end_idx: int) -> Tuple[float, float, int]:
    """
    주어진 [start_idx, end_idx]에서 양쪽 모두 유효한 t들의 교집합을 골라
    trapz 적분. 반환: (baseline_area, proposed_area, used_length)
    유효 t가 없으면 (nan, nan, 0)
    """
    T = b_mean.shape[0]
    start_idx = max(0, min(start_idx, T-1))
    end_idx   = max(0, min(end_idx,   T-1))
    if end_idx < start_idx:
        return float('nan'), float('nan'), 0

    mask = (b_cnt > 0) & (p_cnt > 0)
    idx = np.where(mask & (np.arange(T) >= start_idx) & (np.arange(T) <= end_idx))[0]
    if idx.size == 0:
        return float('nan'), float('nan'), 0

    b_area = float(np.trapz(b_mean[idx], dx=1.0))
    p_area = float(np.trapz(p_mean[idx], dx=1.0))
    return b_area, p_area, int(idx.size)

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # 폴더 입력
    ap.add_argument("--baseline20_dir", type=str,  default='/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/baseline_20', help="baseline .npy가 있는 폴더")
    ap.add_argument("--baseline30_dir", type=str,  default='/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/baseline_30', help="baseline .npy가 있는 폴더")
    ap.add_argument("--proposed20_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/proposed/20', help="proposed .npy가 있는 폴더")
    ap.add_argument("--proposed30_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/proposed/30', help="proposed .npy가 있는 폴더")
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--out_csv", type=str, default="/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/auuc_windows.csv")
    # 옵션
    ap.add_argument("--force_softmax", action="store_true")
    ap.add_argument("--log_base", type=float, default=np.e)
    args = ap.parse_args()

    # 파일 수집
    b20_files = list_sorted_files(args.baseline20_dir, args.pattern)
    b30_files = list_sorted_files(args.baseline30_dir, args.pattern)
    p20_files = list_sorted_files(args.proposed20_dir, args.pattern)
    p30_files = list_sorted_files(args.proposed30_dir, args.pattern)

    if len(b20_files) == 0 or len(p20_files) == 0:
        print("[WARN] baseline_20 또는 proposed_20에 파일이 없습니다.")
    if len(b30_files) == 0 or len(p30_files) == 0:
        print("[WARN] baseline_30 또는 proposed_30에 파일이 없습니다.")

    # ---------- 20% 그룹 ----------
    results = []
    if b20_files and p20_files:
        T_max_20 = max([get_T_len(fp) for fp in (b20_files + p20_files)])
        start_20 = int(np.floor(T_max_20 * 20 / 70))
        ends_20  = [int(np.floor(T_max_20 * r / 70)) for r in (30, 40, 50, 70)]

        b20_mean, b20_cnt = build_group_mean(b20_files, args.force_softmax, args.log_base, T_max_20)
        p20_mean, p20_cnt = build_group_mean(p20_files, args.force_softmax, args.log_base, T_max_20)

        print(f"[20-group] T_max={T_max_20}, start={start_20}, ends={ends_20}")
        for e in ends_20:
            b_area, p_area, L = integrate_over_range(b20_mean, p20_mean, b20_cnt, p20_cnt, start_20, e)
            results.append(["20_group", T_max_20, start_20, e, L, b_area, p_area])

    # ---------- 30% 그룹 ----------
    if b30_files and p30_files:
        T_max_30 = max([get_T_len(fp) for fp in (b30_files + p30_files)])
        start_30 = int(np.floor(T_max_30 * 30 / 80))
        ends_30  = [int(np.floor(T_max_30 * r / 80)) for r in (40, 50, 60, 80)]

        b30_mean, b30_cnt = build_group_mean(b30_files, args.force_softmax, args.log_base, T_max_30)
        p30_mean, p30_cnt = build_group_mean(p30_files, args.force_softmax, args.log_base, T_max_30)

        print(f"[30-group] T_max={T_max_30}, start={start_30}, ends={ends_30}")
        for e in ends_30:
            b_area, p_area, L = integrate_over_range(b30_mean, p30_mean, b30_cnt, p30_cnt, start_30, e)
            results.append(["30_group", T_max_30, start_30, e, L, b_area, p_area])

    # 출력
    if not results:
        raise RuntimeError("적분 결과가 없습니다. 입력 폴더/파일을 확인하세요.")

    # 콘솔 요약
    unit = "nat" if np.isclose(args.log_base, np.e) else ("bit" if np.isclose(args.log_base, 2.0) else f"log{args.log_base}")
    print("\n== AUUC (area under mean expected-entropy) ==")
    print(f"(단위: {unit}·step, L=사용된 유효 t 개수; 적분은 양쪽 유효 t 교집합에서 수행)")
    for grp, Tm, s, e, L, bA, pA in results:
        print(f"[{grp}] T_max={Tm} | range=[{s}..{e}] (L={L}) | Baseline={bA/L:.6f} | Proposed={pA/L:.6f}")

    # CSV 저장
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "T_max", "start_idx", "end_idx", "used_len", "baseline_area", "proposed_area", "unit"])
        for row in results:
            w.writerow(row + [unit])
    print(f"\nSaved CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
