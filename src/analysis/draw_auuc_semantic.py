#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, argparse, csv
from typing import List, Tuple, Dict
import numpy as np

# =============================
# 정렬/유틸
# =============================
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_sorted_files(folder: str, pattern: str="*.npy") -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# =============================
# 엔트로피
# =============================
def expected_entropy_over_S(prob_stc: np.ndarray, log_base: float=np.e, eps: float=1e-12) -> np.ndarray:
    """
    클래스 단위 불확실성: (S,T,C) -> (T,)
    H_st = -∑_c p log_b p,  S축 평균
    """
    if prob_stc.ndim != 3:
        raise ValueError(f"Expected (S,T,C), got {prob_stc.shape}")
    # 안정화 + 재정규화(클립 대신) 권장
    p = np.maximum(prob_stc, eps)
    p = p / np.sum(p, axis=-1, keepdims=True)
    H_st = -np.sum(p * (np.log(p) / np.log(log_base)), axis=-1)  # (S,T)
    return H_st.mean(axis=0)  # (T,)

# =============================
# Semantic grouping
# =============================
# 예시: 필요에 맞게 수정/확장 가능
SEMANTIC_GROUPS: Dict[str, List[int]] = {
    "pancake":           [0,2,3,4,5,6,7,8,14,24,25,27,28,31,32,33,39,40,41,45],
    "coffee":            [1,5,6,15,18,19,20,21,22,23,25,27,28,30,34,38,40,42,44],
    "kitchen_cleaning":  [9,10,11,12,34,40],
    "device":            [13,34,36,37],
    "dining":            [15,17,27,28,31,32,33,34,40],
    "dish_cleaning":     [16,26,34,35,40,43,46],
    # 지정되지 않은 클래스는 'other'로 자동 귀속
}

SEMANTIC_GROUPS_L2 = {
    "cooking" : [0,3,5,6,7,10],
    "cleaning": [1,2,12],
    "dining": [4,9,15],
    "device":[8,11,13,14],
}


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

    # 클래스가 여러 그룹에 속할 수 있을 때 분배 가중치(동등 분배)
    per_class_norm = [1.0/len(g) if len(g)>0 else 0.0 for g in class_to_groups]
    return class_to_groups, per_class_norm, group_names

def classes_to_group_probs(prob_stc: np.ndarray,
                           class_to_groups: List[List[int]],
                           per_class_norm: List[float],
                           num_groups: int) -> np.ndarray:
    """(S,T,C) → (S,T,G)"""
    S, T, C = prob_stc.shape
    G = num_groups
    out = np.zeros((S, T, G), dtype=np.float64)
    for c in range(C):
        if not class_to_groups[c]:
            continue
        share = per_class_norm[c]
        for g in class_to_groups[c]:
            out[..., g] += prob_stc[..., c] * share
    row_sum = out.sum(axis=-1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum

def entropy_over_groups(p_stg: np.ndarray, log_base: float=np.e, eps: float=1e-12) -> np.ndarray:
    """(S,T,G) → (S,T) 그룹 엔트로피"""
    p = np.maximum(p_stg, eps)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return -np.sum(p * (np.log(p) / np.log(log_base)), axis=-1)

def semantic_expected_entropy_over_S(prob_stc: np.ndarray,
                                     log_base: float=np.e,
                                     groups: Dict[str, List[int]]=SEMANTIC_GROUPS_L2) -> np.ndarray:
    """
    Semantic uncertainty: 클래스→그룹 매핑 후 그룹 엔트로피를 S축 평균.
    입력: (S,T,C) 확률
    출력: (T,)
    """
    if prob_stc.ndim != 3:
        raise ValueError(f"Expected (S,T,C), got {prob_stc.shape}")
    S,T,C = prob_stc.shape
    class_to_groups, per_class_norm, group_names = build_class_to_groups(C, groups)
    p_stg = classes_to_group_probs(prob_stc, class_to_groups, per_class_norm, len(group_names))  # (S,T,G)
    H_st = entropy_over_groups(p_stg, log_base=log_base)  # (S,T)
    return H_st.mean(axis=0)  # (T,)

# =============================
# 공통 처리
# =============================
def compute_curve(npy_path: str,
                  force_softmax: bool,
                  log_base: float,
                  use_semantic: bool) -> np.ndarray:
    """
    파일 하나에서 (T,) 곡선 생성:
    - use_semantic=False → 클래스 엔트로피(mean expected entropy)
    - use_semantic=True  → semantic 그룹 엔트로피(mean over S)
    """
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim == 4:   # (S,B,T,C) → B=0
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{npy_path}: expected (S,T,C) or (S,1,T,C), got {arr.shape}")
    prob = arr#ensure_prob(arr, force_softmax=force_softmax)

    if use_semantic:
        return semantic_expected_entropy_over_S(prob, log_base=log_base)  # (T,)
    else:
        return expected_entropy_over_S(prob, log_base=log_base)           # (T,)

def get_T_len(npy_path: str) -> int:
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{npy_path}: expected (S,T,C), got {arr.shape}")
    return arr.shape[1]

def build_group_mean(files: List[str],
                     force_softmax: bool,
                     log_base: float,
                     T_max: int,
                     use_semantic: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    return:
      mean_t: (T_max,) 시간별 평균 (NaN 제외)
      cnt_t : (T_max,) 각 t에서 유효 파일 수
    """
    if len(files) == 0:
        return np.full(T_max, np.nan), np.zeros(T_max, dtype=int)

    mat = np.full((len(files), T_max), np.nan, dtype=np.float64)
    for i, fp in enumerate(files):
        curve = compute_curve(fp, force_softmax, log_base, use_semantic)  # (T_i,)
        Ti = min(len(curve), T_max)
        if Ti > 0:
            mat[i, :Ti] = curve[:Ti]

    mean_t = np.nanmean(mat, axis=0)
    cnt_t  = np.sum(~np.isnan(mat), axis=0).astype(int)
    return mean_t, cnt_t

def integrate_over_range(b_mean: np.ndarray, p_mean: np.ndarray,
                         b_cnt: np.ndarray, p_cnt: np.ndarray,
                         start_idx: int, end_idx: int) -> Tuple[float, float, int]:
    """
    [start_idx, end_idx] 교집합 유효 t에서 trapz 적분
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

# =============================
# 메인
# =============================
def main():
    ap = argparse.ArgumentParser()
    # 폴더 입력
    ap.add_argument("--baseline20_dir", type=str,  default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/baseline_20')
    ap.add_argument("--baseline30_dir", type=str,  default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/baseline_30')
    ap.add_argument("--proposed20_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/proposed/20')
    ap.add_argument("--proposed30_dir", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/proposed/30')
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--out_csv", type=str, default="/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/auuc_windows_semantic.csv")
    # 옵션
    ap.add_argument("--force_softmax", action="store_true", help="입력이 로짓/비정규화이면 강제 softmax")
    ap.add_argument("--log_base", type=float, default=np.e, help="로그 밑 (e 또는 2 권장)")
    ap.add_argument("--use_semantic", default="true", help="Semantic grouping 기반 불확실성 사용")
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

    results = []

    # ---------- 20% 그룹 ----------
    if b20_files and p20_files:
        T_max_20 = max([get_T_len(fp) for fp in (b20_files + p20_files)])
        start_20 = int(np.floor(T_max_20 * 20 / 70))
        ends_20  = [int(np.floor(T_max_20 * r / 70)) for r in (30, 40, 50, 70)]

        b20_mean, b20_cnt = build_group_mean(b20_files, args.force_softmax, args.log_base, T_max_20, args.use_semantic)
        p20_mean, p20_cnt = build_group_mean(p20_files, args.force_softmax, args.log_base, T_max_20, args.use_semantic)

        print(f"[20-group] T_max={T_max_20}, start={start_20}, ends={ends_20}")
        for e in ends_20:
            b_area, p_area, L = integrate_over_range(b20_mean, p20_mean, b20_cnt, p20_cnt, start_20, e)
            results.append(["20_group", T_max_20, start_20, e, L, b_area, p_area])

    # ---------- 30% 그룹 ----------
    if b30_files and p30_files:
        T_max_30 = max([get_T_len(fp) for fp in (b30_files + p30_files)])
        start_30 = int(np.floor(T_max_30 * 30 / 80))
        ends_30  = [int(np.floor(T_max_30 * r / 80)) for r in (40, 50, 60, 80)]

        b30_mean, b30_cnt = build_group_mean(b30_files, args.force_softmax, args.log_base, T_max_30, args.use_semantic)
        p30_mean, p30_cnt = build_group_mean(p30_files, args.force_softmax, args.log_base, T_max_30, args.use_semantic)

        print(f"[30-group] T_max={T_max_30}, start={start_30}, ends={ends_30}")
        for e in ends_30:
            b_area, p_area, L = integrate_over_range(b30_mean, p30_mean, b30_cnt, p30_cnt, start_30, e)
            results.append(["30_group", T_max_30, start_30, e, L, b_area, p_area])

    # 출력
    if not results:
        raise RuntimeError("적분 결과가 없습니다. 입력 폴더/파일을 확인하세요.")

    unit = "nat" if np.isclose(args.log_base, np.e) else ("bit" if np.isclose(args.log_base, 2.0) else f"log{args.log_base}")
    kind = "Semantic Expected Entropy" if args.use_semantic else "Class Expected Entropy"
    print(f"\n== AUUC (area under mean {kind}) ==")
    print(f"(단위: {unit}·step, L=사용된 유효 t 개수; 적분은 양쪽 유효 t 교집합에서 수행)")
    for grp, Tm, s, e, L, bA, pA in results:
        # 평균 면적(정규화)도 빠르게 확인하려면 bA/L, pA/L 사용
        print(f"[{grp}] T_max={Tm} | range=[{s}..{e}] (L={L}) | Baseline={bA:.6f} | Proposed={pA:.6f} | mean/base={bA/L:.6f} | mean/prop={pA/L:.6f}")

    # CSV 저장
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "T_max", "start_idx", "end_idx", "used_len", "baseline_area", "proposed_area", "unit", "kind"])
        for row in results:
            w.writerow(row + [unit, kind])
    print(f"\nSaved CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
