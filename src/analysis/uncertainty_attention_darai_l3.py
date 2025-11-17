# # import os, glob, re, json
# # from typing import List, Dict, Tuple
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # -----------------------------
# # # 자연 정렬
# # # -----------------------------
# # def _natural_key(s: str):
# #     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# # def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
# #     paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
# #     return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# # # -----------------------------
# # # 의미군 정의
# # # -----------------------------
# # SEMANTIC_GROUPS = {
# #     "pancake":           [0,2,3,4,5,6,7,8,14,24,25,27,28,31,32,33,39,40,41,45],
# #     "coffee":            [1,5,6,15,18,19,20,21,22,23,25,27,28,30,34,38,40,42,44],
# #     "kitchen_cleaning":  [9,10,11,12,34,40],
# #     "device":            [13,34,36,37],
# #     "dining":            [15,17,27,28,31,32,33,34,40],
# #     "dish_cleaning":     [16,26,34,35,40,43,46],
# # }

# # # -----------------------------
# # # 유틸
# # # -----------------------------
# # def sanitize(arr: np.ndarray) -> np.ndarray:
# #     arr = np.array(arr, dtype=np.float64, copy=True)
# #     arr[~np.isfinite(arr)] = 0.0
# #     return arr

# # def looks_prob(arr: np.ndarray, tol=1e-3) -> bool:
# #     if arr.ndim < 1: return False
# #     row_sum = np.nansum(arr, axis=-1)
# #     return np.all((row_sum > 1 - tol) & (row_sum < 1 + tol)) and np.nanmin(arr) >= -tol

# # def softmax_last(x: np.ndarray) -> np.ndarray:
# #     x = x - np.nanmax(x, axis=-1, keepdims=True)
# #     ex = np.exp(x)
# #     s = np.nansum(ex, axis=-1, keepdims=True)
# #     s[s == 0] = 1.0
# #     return ex / s

# # def ensure_prob(arr: np.ndarray) -> np.ndarray:
# #     return arr if looks_prob(arr) else softmax_last(arr)

# # def entropy_over_groups(p: np.ndarray, eps=1e-12) -> np.ndarray:
# #     p = np.clip(p, eps, 1.0)
# #     return -np.sum(p * np.log(p), axis=-1)

# # def build_class_to_groups(num_classes: int, group_dict: Dict[str, List[int]]):
# #     group_names = list(group_dict.keys())
# #     explicit = {k: set(v) for k, v in group_dict.items()}
# #     assigned = set().union(*explicit.values()) if explicit else set()
# #     need_other = len(assigned) < num_classes
# #     if need_other:
# #         group_names.append("other")
# #     name2idx = {n:i for i,n in enumerate(group_names)}

# #     class_to_groups = [[] for _ in range(num_classes)]
# #     for name, cls_set in explicit.items():
# #         gi = name2idx[name]
# #         for c in cls_set:
# #             if 0 <= c < num_classes:
# #                 class_to_groups[c].append(gi)

# #     if need_other:
# #         other_idx = name2idx["other"]
# #         for c in range(num_classes):
# #             if not class_to_groups[c]:
# #                 class_to_groups[c].append(other_idx)

# #     per_class_norm = [1.0/len(g) if len(g)>0 else 0.0 for g in class_to_groups]
# #     return class_to_groups, per_class_norm, group_names

# # def classes_to_group_probs(prob_stc: np.ndarray,
# #                            class_to_groups: List[List[int]],
# #                            per_class_norm: List[float],
# #                            num_groups: int) -> np.ndarray:
# #     S,T,C = prob_stc.shape
# #     out = np.zeros((S,T,num_groups), dtype=np.float64)
# #     for c in range(C):
# #         if not class_to_groups[c]:
# #             continue
# #         share = per_class_norm[c]
# #         for g in class_to_groups[c]:
# #             out[..., g] += prob_stc[..., c] * share
# #     row_sum = out.sum(axis=-1, keepdims=True)
# #     row_sum[row_sum==0] = 1.0
# #     return out / row_sum

# # # -----------------------------
# # # Attention 로딩/정규화
# # # -----------------------------
# # def load_attention_1d(attn_path: str) -> np.ndarray:
# #     """
# #     지원 형태:
# #       (B, 1, T)  -> 평균(B) -> (T,)
# #       (1, T)     -> (T,)
# #       (H, T)     -> 평균(H) -> (T,)
# #       (T,)       -> (T,)
# #     """
# #     a = np.load(attn_path, allow_pickle=False)
# #     a = sanitize(a)
# #     if a.ndim == 3 and a.shape[1] == 1:
# #         a = a.mean(axis=0)[0]   # (T,)
# #     elif a.ndim == 2:
# #         a = a.mean(axis=0)      # (T,)
# #     elif a.ndim == 1:
# #         pass
# #     else:
# #         raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")

# #     # 정규화(면적=1)
# #     s = a.sum()
# #     if s != 0:
# #         a = a / s
# #     return a  # (T,)

# # # -----------------------------
# # # 파일 단위 처리
# # # -----------------------------
# # def compute_time_uncertainty(stc_path: str) -> Tuple[np.ndarray, Dict]:
# #     """
# #     (S,T,C) -> 시간별 semantic uncertainty (T,)
# #     """
# #     x = np.load(stc_path, allow_pickle=False)
# #     if x.ndim == 4:
# #         x = x[:, :1, :, :].squeeze(1)
# #     elif x.ndim != 3:
# #         raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C), got {x.shape}")
    
# #     S,T,C = x.shape
# #     probs = ensure_prob(sanitize(x))
# #     class_to_groups, per_class_norm, group_names = build_class_to_groups(C, SEMANTIC_GROUPS)
# #     stg = classes_to_group_probs(probs, class_to_groups, per_class_norm, len(group_names))
# #     H_st = entropy_over_groups(stg)  # (S,T)
# #     H_t = H_st.mean(axis=0)          # (T,)
# #     return H_t, {"S":S, "T":T, "C":C, "groups":group_names}

# # def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
# #     if a.size != b.size:
# #         m = min(a.size, b.size)
# #         a, b = a[:m], b[:m]
# #     if np.std(a) < 1e-12 or np.std(b) < 1e-12:
# #         return 0.0
# #     return float(np.corrcoef(a, b)[0,1])

# # def attention_coverage_on_topq(attn: np.ndarray, unc: np.ndarray, q=0.1) -> float:
# #     """
# #     상위 q 비율(불확실성 높은 타임스텝 집합)에서 어텐션이 얼마나 쏠렸는지.
# #     attn은 sum=1로 정규화되어 있음.
# #     """
# #     T = min(attn.size, unc.size)
# #     attn = attn[:T]; unc = unc[:T]
# #     k = max(1, int(round(T * q)))
# #     idx = np.argpartition(-unc, k-1)[:k]  # top-k indices (unordered)
# #     return float(attn[idx].sum())

# # def plot_unc_vs_attn(unc_t: np.ndarray, attn_t: np.ndarray, title: str, out_path: str):
# #     T = min(unc_t.size, attn_t.size)
# #     x = np.arange(T)
# #     fig, ax1 = plt.subplots(figsize=(12, 3.2))
# #     ax1.plot(x, unc_t[:T], label="Semantic Uncertainty (H)", linewidth=2)
# #     ax1.set_xlabel("Time")
# #     ax1.set_ylabel("Entropy (H)", color="C0")
# #     ax2 = ax1.twinx()
# #     ax2.plot(x, attn_t[:T], label="Attention", linewidth=1.5, alpha=0.8)
# #     ax2.set_ylabel("Attention (normalized)", color="C1")
# #     ax1.set_title(title)
# #     fig.tight_layout()
# #     fig.savefig(out_path, dpi=200)
# #     plt.close(fig)

# # # -----------------------------
# # # 폴더 비교 파이프라인
# # # -----------------------------
# # def process_folder(model_name: str,
# #                    stc_dir: str,
# #                    attn_dir: str,
# #                    out_root: str,
# #                    pattern: str = "*.npy"):
# #     """
# #     model_name: "baseline" 또는 "infer_goal"
# #     stc_dir: (S,T,C) 파일 폴더
# #     attn_dir: attention npy 폴더
# #     """
# #     out_dir = os.path.join(out_root, model_name)
# #     os.makedirs(out_dir, exist_ok=True)

# #     stc_files = list_sorted_files(stc_dir, pattern)
# #     if not stc_files:
# #         raise RuntimeError(f"No files in {stc_dir}")

# #     rows = []
# #     for i, stc_path in enumerate(stc_files):
# #         stem = os.path.splitext(os.path.basename(stc_path))[0]
# #         # attention 파일 매칭: 같은 스템 이름 우선
# #         candidate = os.path.join(attn_dir, f"{stem}.npy")
# #         if not os.path.isfile(candidate):
# #             # fallback: 자연 정렬로 attention 폴더에서 같은 인덱스 파일 매칭(권장하지 않지만 보조)
# #             attn_list = list_sorted_files(attn_dir, pattern)
# #             if i < len(attn_list):
# #                 candidate = attn_list[i]
# #             else:
# #                 print(f"[{model_name}] WARN: attention not found for {stem}, skip.")
# #                 continue

# #         unc_t, meta = compute_time_uncertainty(stc_path)
# #         attn_t = load_attention_1d(candidate)

# #         r = pearson_corr(unc_t, attn_t)
# #         cov10 = attention_coverage_on_topq(attn_t, unc_t, q=0.10)
# #         cov20 = attention_coverage_on_topq(attn_t, unc_t, q=0.20)

# #         # 저장: 플롯
# #         png_path = os.path.join(out_dir, f"{stem}_unc_vs_attn.png")
# #         plot_unc_vs_attn(unc_t, attn_t, f"{model_name}: {stem}", png_path)

# #         rows.append({
# #             "idx": i,
# #             "file": os.path.basename(stc_path),
# #             "attn_file": os.path.basename(candidate),
# #             "T_used": int(min(unc_t.size, attn_t.size)),
# #             "pearson_r": r,
# #             "coverage_top10pct": cov10,
# #             "coverage_top20pct": cov20,
# #             "S": meta["S"], "T": meta["T"], "C": meta["C"]
# #         })
# #         print(f"[{model_name}][{i:03d}] {stem}: r={r:.3f}, cov@10%={cov10:.3f}, cov@20%={cov20:.3f}")
        

# #     df = pd.DataFrame(rows)
# #     csv_path = os.path.join(out_dir, "summary.csv")
# #     df.to_csv(csv_path, index=False)

# #     # 집계
# #     agg = {}
# #     if not df.empty:
# #         agg = {
# #             "model": model_name,
# #             "num_files": int(len(df)),
# #             "mean_r": float(df["pearson_r"].mean()),
# #             "std_r": float(df["pearson_r"].std(ddof=1)) if len(df) > 1 else 0.0,
# #             "mean_cov10": float(df["coverage_top10pct"].mean()),
# #             "mean_cov20": float(df["coverage_top20pct"].mean()),
# #             "summary_csv": csv_path,
# #             "out_dir": out_dir
# #         }
# #         with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
# #             json.dump(agg, f, indent=2)
# #     return df, agg

# # # -----------------------------
# # # 메인
# # # -----------------------------
# # if __name__ == "__main__":
# #     # 경로 설정
# #     BASE_DIR = "/home/hice1/skim3513/scratch/causdiff/outputs"
# #     STC_BASELINE = os.path.join(BASE_DIR, "baselines_20")
# #     STC_INFERGOAL = os.path.join(BASE_DIR, "infer_goal", "20")
# #     ATTN_DIR = os.path.join(BASE_DIR, "attention_map_20")
# #     OUT_ROOT = "./uncertainty_attention_out"

# #     os.makedirs(OUT_ROOT, exist_ok=True)

# #     # 베이스라인 처리
# #     df_b, agg_b = process_folder("baseline", STC_BASELINE, ATTN_DIR, OUT_ROOT)

# #     # infer_goal 처리
# #     df_g, agg_g = process_folder("infer_goal", STC_INFERGOAL, ATTN_DIR, OUT_ROOT)

# #     # 두 모델 비교 집계 저장
# #     compare = {
# #         "baseline": agg_b,
# #         "infer_goal": agg_g
# #     }
# #     with open(os.path.join(OUT_ROOT, "compare_aggregate.json"), "w") as f:
# #         json.dump(compare, f, indent=2)

# #     print("\n=== Aggregate ===")
# #     print(json.dumps(compare, indent=2))
# import os, glob, re, json
# from typing import List, Dict, Tuple
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # 자연 정렬
# # -----------------------------
# def _natural_key(s: str):
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
#     paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
#     return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# # -----------------------------
# # 의미군 정의 (baseline 전용)
# # -----------------------------
# SEMANTIC_GROUPS = {
#     "pancake":           [0,2,3,4,5,6,7,8,14,24,25,27,28,31,32,33,39,40,41,45],
#     "coffee":            [1,5,6,15,18,19,20,21,22,23,25,27,28,30,34,38,40,42,44],
#     "kitchen_cleaning":  [9,10,11,12,34,40],
#     "device":            [13,34,36,37],
#     "dining":            [15,17,27,28,31,32,33,34,40],
#     "dish_cleaning":     [16,26,34,35,40,43,46],
# }

# # -----------------------------
# # 유틸
# # -----------------------------
# def sanitize(arr: np.ndarray) -> np.ndarray:
#     arr = np.array(arr, dtype=np.float64, copy=True)
#     arr[~np.isfinite(arr)] = 0.0
#     return arr

# def looks_prob(arr: np.ndarray, tol=1e-3) -> bool:
#     if arr.ndim < 1: return False
#     row_sum = np.nansum(arr, axis=-1)
#     return np.all((row_sum > 1 - tol) & (row_sum < 1 + tol)) and np.nanmin(arr) >= -tol

# def softmax_last(x: np.ndarray) -> np.ndarray:
#     x = x - np.nanmax(x, axis=-1, keepdims=True)
#     ex = np.exp(x)
#     s = np.nansum(ex, axis=-1, keepdims=True)
#     s[s == 0] = 1.0
#     return ex / s

# def ensure_prob(arr: np.ndarray) -> np.ndarray:
#     # 로짓/피처면 softmax, 이미 확률이면 그대로
#     return arr if looks_prob(arr) else softmax_last(arr)

# def entropy_over_groups(p: np.ndarray, eps=1e-12) -> np.ndarray:
#     # 마지막 축에 대해 엔트로피 계산 (그룹/클래스 상관없이 동일하게 동작)
#     p = np.clip(p, eps, 1.0)
#     return -np.sum(p * np.log(p), axis=-1)

# def build_class_to_groups(num_classes: int, group_dict: Dict[str, List[int]]):
#     group_names = list(group_dict.keys())
#     explicit = {k: set(v) for k, v in group_dict.items()}
#     assigned = set().union(*explicit.values()) if explicit else set()
#     need_other = len(assigned) < num_classes
#     if need_other:
#         group_names.append("other")
#     name2idx = {n:i for i,n in enumerate(group_names)}

#     class_to_groups = [[] for _ in range(num_classes)]
#     for name, cls_set in explicit.items():
#         gi = name2idx[name]
#         for c in cls_set:
#             if 0 <= c < num_classes:
#                 class_to_groups[c].append(gi)

#     if need_other:
#         other_idx = name2idx["other"]
#         for c in range(num_classes):
#             if not class_to_groups[c]:
#                 class_to_groups[c].append(other_idx)

#     per_class_norm = [1.0/len(g) if len(g)>0 else 0.0 for g in class_to_groups]
#     return class_to_groups, per_class_norm, group_names

# def classes_to_group_probs(prob_stc: np.ndarray,
#                            class_to_groups: List[List[int]],
#                            per_class_norm: List[float],
#                            num_groups: int) -> np.ndarray:
#     S,T,C = prob_stc.shape
#     out = np.zeros((S,T,num_groups), dtype=np.float64)
#     for c in range(C):
#         if not class_to_groups[c]:
#             continue
#         share = per_class_norm[c]
#         for g in class_to_groups[c]:
#             out[..., g] += prob_stc[..., c] * share
#     row_sum = out.sum(axis=-1, keepdims=True)
#     row_sum[row_sum==0] = 1.0
#     return out / row_sum

# # -----------------------------
# # Attention 로딩/정규화
# # -----------------------------
# def load_attention_1d(attn_path: str) -> np.ndarray:
#     """
#     지원 형태:
#       (B, 1, T)  -> 평균(B) -> (T,)
#       (1, T)     -> (T,)
#       (H, T)     -> 평균(H) -> (T,)
#       (T,)       -> (T,)
#     """
#     a = np.load(attn_path, allow_pickle=False)
#     a = sanitize(a)
#     if a.ndim == 3 and a.shape[1] == 1:
#         a = a.mean(axis=0)[0]   # (T,)
#     elif a.ndim == 2:
#         a = a.mean(axis=0)      # (T,)
#     elif a.ndim == 1:
#         pass
#     else:
#         raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")

#     # 정규화(면적=1)
#     s = a.sum()
#     if s != 0:
#         a = a / s
#     return a  # (T,)

# # -----------------------------
# # 파일 단위 처리
# # -----------------------------
# def compute_time_uncertainty(stc_path: str, use_groups: bool) -> Tuple[np.ndarray, Dict]:
#     """
#     (S,T,C)[또는 (S,1,T,C)] -> 시간별 불확실성 (T,)
#     - use_groups=True  : baseline 모드 (클래스→그룹 매핑 후 그룹 엔트로피)
#     - use_groups=False : infer_goal 모드 (그룹 없이 클래스 확률로 직접 엔트로피)
#     """
#     x = np.load(stc_path, allow_pickle=False)
#     if x.ndim == 4:
#         # (S,1,T,C) 같은 케이스는 (S,T,C)로 변환
#         x = x[:, 0, :, :]
#     elif x.ndim != 3:
#         raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C), got {x.shape}")

#     S,T,C = x.shape
#     probs = ensure_prob(sanitize(x))

#     if use_groups:
#         class_to_groups, per_class_norm, group_names = build_class_to_groups(C, SEMANTIC_GROUPS)
#         stg = classes_to_group_probs(probs, class_to_groups, per_class_norm, len(group_names))  # (S,T,G)
#         H_st = entropy_over_groups(stg)   # (S,T)
#         H_t  = H_st.mean(axis=0)          # (T,)
#         meta = {"S":S, "T":T, "C":C, "groups":group_names}
#     else:
#         # 그룹 매핑 없이 클래스 분포로 엔트로피 (infer_goal)
#         H_st = entropy_over_groups(probs) # (S,T), C기반
#         H_t  = H_st.mean(axis=0)          # (T,)
#         meta = {"S":S, "T":T, "C":C, "groups": None}
#     return H_t, meta

# def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
#     if a.size != b.size:
#         m = min(a.size, b.size)
#         a, b = a[:m], b[:m]
#     if np.std(a) < 1e-12 or np.std(b) < 1e-12:
#         return 0.0
#     return float(np.corrcoef(a, b)[0,1])

# def attention_coverage_on_topq(attn: np.ndarray, unc: np.ndarray, q=0.1) -> float:
#     """
#     상위 q 비율(불확실성 높은 타임스텝 집합)에서 어텐션이 얼마나 쏠렸는지.
#     attn은 sum=1로 정규화되어 있음.
#     """
#     T = min(attn.size, unc.size)
#     attn = attn[:T]; unc = unc[:T]
#     k = max(1, int(round(T * q)))
#     idx = np.argpartition(-unc, k-1)[:k]  # top-k indices (unordered)
#     return float(attn[idx].sum())

# def plot_unc_vs_attn(unc_t: np.ndarray, attn_t: np.ndarray, title: str, out_path: str):
#     T = min(unc_t.size, attn_t.size)
#     x = np.arange(T)
#     fig, ax1 = plt.subplots(figsize=(12, 3.2))
#     ax1.plot(x, unc_t[:T], label="Semantic Uncertainty (H)", linewidth=2)
#     ax1.set_xlabel("Time")
#     ax1.set_ylabel("Entropy (H)", color="C0")
#     ax2 = ax1.twinx()
#     ax2.plot(x, attn_t[:T], label="Attention", linewidth=1.5, alpha=0.8)
#     ax2.set_ylabel("Attention (normalized)", color="C1")
#     ax1.set_title(title)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)

# # -----------------------------
# # 폴더 비교 파이프라인
# # -----------------------------
# def process_folder(model_name: str,
#                    stc_dir: str,
#                    attn_dir: str,
#                    out_root: str,
#                    pattern: str = "*.npy",
#                    use_groups: bool = True):
#     """
#     model_name: "baseline" 또는 "infer_goal"
#     stc_dir: (S,T,C) 파일 폴더
#     attn_dir: attention npy 폴더
#     use_groups: baseline=True, infer_goal=False
#     """
#     out_dir = os.path.join(out_root, model_name)
#     os.makedirs(out_dir, exist_ok=True)

#     stc_files = list_sorted_files(stc_dir, pattern)
#     if not stc_files:
#         raise RuntimeError(f"No files in {stc_dir}")

#     rows = []
#     attn_list = list_sorted_files(attn_dir, pattern)
#     for i, stc_path in enumerate(stc_files):
#         stem = os.path.splitext(os.path.basename(stc_path))[0]
#         # attention 파일 매칭: 같은 스템 이름 우선
#         candidate = os.path.join(attn_dir, f"{stem}.npy")
#         if not os.path.isfile(candidate):
#             # fallback: 정렬 인덱스 매칭 (가능하면 피하되 보조로 유지)
#             if i < len(attn_list):
#                 candidate = attn_list[i]
#             else:
#                 print(f"[{model_name}] WARN: attention not found for {stem}, skip.")
#                 continue

#         unc_t, meta = compute_time_uncertainty(stc_path, use_groups=use_groups)
#         attn_t = load_attention_1d(candidate)

#         r = pearson_corr(unc_t, attn_t)
#         cov10 = attention_coverage_on_topq(attn_t, unc_t, q=0.10)
#         cov20 = attention_coverage_on_topq(attn_t, unc_t, q=0.20)

#         # 저장: 플롯
#         png_path = os.path.join(out_dir, f"{stem}_unc_vs_attn.png")
#         plot_unc_vs_attn(unc_t, attn_t, f"{model_name}: {stem}", png_path)

#         rows.append({
#             "idx": i,
#             "file": os.path.basename(stc_path),
#             "attn_file": os.path.basename(candidate),
#             "T_used": int(min(unc_t.size, attn_t.size)),
#             "pearson_r": r,
#             "coverage_top10pct": cov10,
#             "coverage_top20pct": cov20,
#             "S": meta["S"], "T": meta["T"], "C": meta["C"]
#         })
#         print(f"[{model_name}][{i:03d}] {stem}: r={r:.3f}, cov@10%={cov10:.3f}, cov@20%={cov20:.3f}")

#     df = pd.DataFrame(rows)
#     csv_path = os.path.join(out_dir, "summary.csv")
#     df.to_csv(csv_path, index=False)

#     # 집계
#     agg = {}
#     if not df.empty:
#         agg = {
#             "model": model_name,
#             "num_files": int(len(df)),
#             "mean_r": float(df["pearson_r"].mean()),
#             "std_r": float(df["pearson_r"].std(ddof=1)) if len(df) > 1 else 0.0,
#             "mean_cov10": float(df["coverage_top10pct"].mean()),
#             "mean_cov20": float(df["coverage_top20pct"].mean()),
#             "summary_csv": csv_path,
#             "out_dir": out_dir
#         }
#         with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
#             json.dump(agg, f, indent=2)
#     return df, agg

# # -----------------------------
# # 메인
# # -----------------------------
# if __name__ == "__main__":
#     BASE_DIR = "/home/hice1/skim3513/scratch/causdiff/outputs"
#     STC_BASELINE = os.path.join(BASE_DIR, "baselines_20")
#     STC_INFERGOAL = os.path.join(BASE_DIR, "infer_goal", "20")
#     ATTN_DIR = os.path.join(BASE_DIR, "attention_map_20")
#     OUT_ROOT = "./uncertainty_attention_out"

#     os.makedirs(OUT_ROOT, exist_ok=True)

#     # baseline: 그룹 매핑 사용
#     df_b, agg_b = process_folder("baseline", STC_BASELINE, ATTN_DIR, OUT_ROOT, use_groups=True)

#     # infer_goal: 그룹 매핑 미사용 (그냥 클래스 기반 엔트로피)
#     df_g, agg_g = process_folder("infer_goal", STC_INFERGOAL, ATTN_DIR, OUT_ROOT, use_groups=False)

#     compare = {"baseline": agg_b, "infer_goal": agg_g}
#     with open(os.path.join(OUT_ROOT, "compare_aggregate.json"), "w") as f:
#         json.dump(compare, f, indent=2)

#     print("\n=== Aggregate ===")
#     print(json.dumps(compare, indent=2))
# save as: compare_uncertainty_with_shared_attention.py
import os, glob, re, json
from typing import List, Dict, Tuple
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

def to_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

# -----------------------------
# 확률/엔트로피 유틸
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
    # 마지막 축 엔트로피
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

# -----------------------------
# baseline 전용: 클래스→의미군 매핑
# -----------------------------
SEMANTIC_GROUPS = {
    "pancake":           [0,2,3,4,5,6,7,8,14,24,25,27,28,31,32,33,39,40,41,45],
    "coffee":            [1,5,6,15,18,19,20,21,22,23,25,27,28,30,34,38,40,42,44],
    "kitchen_cleaning":  [9,10,11,12,34,40],
    "device":            [13,34,36,37],
    "dining":            [15,17,27,28,31,32,33,34,40],
    "dish_cleaning":     [16,26,34,35,40,43,46],
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

    per_class_norm = [1.0/len(g) if len(g)>0 else 0.0 for g in class_to_groups]
    return class_to_groups, per_class_norm, group_names

def classes_to_group_probs(prob_stc: np.ndarray,
                           class_to_groups: List[List[int]],
                           per_class_norm: List[float],
                           num_groups: int) -> np.ndarray:
    S,T,C = prob_stc.shape
    out = np.zeros((S,T,num_groups), dtype=np.float64)
    for c in range(C):
        if not class_to_groups[c]:
            continue
        share = per_class_norm[c]
        for g in class_to_groups[c]:
            out[..., g] += prob_stc[..., c] * share
    row_sum = out.sum(axis=-1, keepdims=True)
    row_sum[row_sum==0] = 1.0
    return out / row_sum

# -----------------------------
# 어텐션 로딩(공통)
# -----------------------------
def load_attention_1d(attn_path: str) -> np.ndarray:
    a = np.load(attn_path, allow_pickle=False)
    a = sanitize(a)
    if a.ndim == 3 and a.shape[1] == 1:
        a = a.mean(axis=0)[0]   # (T,)
    elif a.ndim == 2:
        a = a.mean(axis=0)      # (T,)
    elif a.ndim == 1:
        pass
    else:
        raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")
    s = a.sum()
    if s != 0: a = a / s
    return a

# -----------------------------
# 불확실성 계산기
# -----------------------------
def time_unc_baseline(stc_path: str) -> Tuple[np.ndarray, Dict]:
    x = np.load(stc_path, allow_pickle=False)
    if x.ndim == 4:
        x = x[:,0,:,:]
    elif x.ndim != 3:
        raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C), got {x.shape}")
    S,T,C = x.shape
    probs = ensure_prob(sanitize(x))
    ctg, w, gnames = build_class_to_groups(C, SEMANTIC_GROUPS)
    stg = classes_to_group_probs(probs, ctg, w, len(gnames))
    H_st = entropy_over_last(stg)
    return H_st.mean(axis=0), {"S":S,"T":T,"C":C}

def time_unc_infergoal(stc_path: str) -> Tuple[np.ndarray, Dict]:
    x = np.load(stc_path, allow_pickle=False)
    if x.ndim == 4:
        x = x[:,0,:,:]
    elif x.ndim != 3:
        raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C), got {x.shape}")
    S,T,C = x.shape
    probs = ensure_prob(sanitize(x))  # 피처/로짓이면 softmax
    H_st = entropy_over_last(probs)
    return H_st.mean(axis=0), {"S":S,"T":T,"C":C}

# -----------------------------
# 상관/커버리지 & 플롯
# -----------------------------
def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0,1])

def attention_coverage_on_topq(attn: np.ndarray, unc: np.ndarray, q=0.1) -> float:
    T = min(attn.size, unc.size)
    attn = attn[:T]; unc = unc[:T]
    k = max(1, int(round(T * q)))
    idx = np.argpartition(-unc, k-1)[:k]
    return float(attn[idx].sum())

def plot_unc_vs_attn(unc_t: np.ndarray, attn_t: np.ndarray, title: str, out_path: str):
    T = min(unc_t.size, attn_t.size)
    x = np.arange(T)
    fig, ax1 = plt.subplots(figsize=(12, 3.2))
    ax1.plot(x, unc_t[:T], linewidth=2)
    ax1.set_xlabel("Time"); ax1.set_ylabel("Entropy (H)", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(x, attn_t[:T], linewidth=1.3, alpha=0.9)
    ax2.set_ylabel("Attention (normalized)", color="C1")
    ax1.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def run_with_shared_attention(
    baseline_dir: str,
    infergoal_dir: str,
    attn_dir: str,
    out_root: str,
    pattern: str = "*.npy"
):
    os.makedirs(out_root, exist_ok=True)
    b_files = list_sorted_files(baseline_dir, pattern)
    g_files = list_sorted_files(infergoal_dir, pattern)
    a_files = list_sorted_files(attn_dir, pattern)

    if not b_files or not g_files or not a_files:
        raise RuntimeError("One or more folders are empty.")

    # 인덱스 기준 최소 길이만큼만 비교
    N = min(len(b_files), len(g_files), len(a_files))
    if len({len(b_files), len(g_files), len(a_files)}) != 1:
        print(f"[WARN] count mismatch: baseline={len(b_files)}, infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

    out_b = os.path.join(out_root, "baseline");   os.makedirs(out_b, exist_ok=True)
    out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

    rows = []
    print(f"Using shared attention from: {attn_dir}")
    print(f"#pairs (by index) = {N}")

    for i in range(N):
        b_path = b_files[i]
        g_path = g_files[i]
        a_path = a_files[i]
        stem   = os.path.splitext(os.path.basename(b_path))[0]  # 로그/파일명 표기를 위해

        # 공통 attention
        attn_t = load_attention_1d(a_path)

        # baseline uncertainty
        unc_b, meta_b = time_unc_baseline(b_path)
        r_b = pearson_corr(unc_b, attn_t)
        cov10_b = attention_coverage_on_topq(attn_t, unc_b, 0.10)
        cov20_b = attention_coverage_on_topq(attn_t, unc_b, 0.20)
        plot_unc_vs_attn(unc_b, attn_t,
                         f"baseline: {stem}",
                         os.path.join(out_b, f"{stem}_unc_vs_attn.png"))

        # infer_goal uncertainty (그룹 매핑 없이)
        unc_g, meta_g = time_unc_infergoal(g_path)
        r_g = pearson_corr(unc_g, attn_t)
        cov10_g = attention_coverage_on_topq(attn_t, unc_g, 0.10)
        cov20_g = attention_coverage_on_topq(attn_t, unc_g, 0.20)
        plot_unc_vs_attn(unc_g, attn_t,
                         f"infer_goal: {stem}",
                         os.path.join(out_g, f"{stem}_unc_vs_attn.png"))

        rows.append({
            "idx": i,
            "baseline_file": os.path.basename(b_path),
            "infer_goal_file": os.path.basename(g_path),
            "attention_file": os.path.basename(a_path),
            "r_baseline": r_b, "cov10_baseline": cov10_b, "cov20_baseline": cov20_b,
            "S_b": meta_b["S"], "T_b": meta_b["T"], "C_b": meta_b["C"],
            "r_infer_goal": r_g, "cov10_infer_goal": cov10_g, "cov20_infer_goal": cov20_g,
            "S_g": meta_g["S"], "T_g": meta_g["T"], "C_g": meta_g["C"],
        })
        print(f"[{i:03d}] r_b={r_b:.3f}, r_g={r_g:.3f}, cov10(b/g)={cov10_b:.3f}/{cov10_g:.3f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "summary_shared_attention_by_index.csv")
    df.to_csv(csv_path, index=False)

    agg = {
        "num_files": int(len(df)),
        "baseline": {
            "mean_r": float(df["r_baseline"].mean()),
            "std_r": float(df["r_baseline"].std(ddof=1)) if len(df) > 1 else 0.0,
            "mean_cov10": float(df["cov10_baseline"].mean()),
            "mean_cov20": float(df["cov20_baseline"].mean()),
        },
        "infer_goal": {
            "mean_r": float(df["r_infer_goal"].mean()),
            "std_r": float(df["r_infer_goal"].std(ddof=1)) if len(df) > 1 else 0.0,
            "mean_cov10": float(df["cov10_infer_goal"].mean()),
            "mean_cov20": float(df["cov20_infer_goal"].mean()),
        },
        "summary_csv": csv_path,
        "out_root": out_root
    }
    with open(os.path.join(out_root, "aggregate_shared_attention_by_index.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== Aggregate (shared attention; by index) ===")
    print(json.dumps(agg, indent=2))


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    BASE = "/home/hice1/skim3513/scratch/causdiff/outputs"
    BASELINE_DIR  = os.path.join(BASE, "baselines_30")
    INFERGOAL_DIR = os.path.join(BASE, "infer_goal", "30")
    ATTENTION_DIR = os.path.join(BASE, "attention_map_30")
    OUT = "./uncertainty_attention_out_shared"

    run_with_shared_attention(
        baseline_dir=BASELINE_DIR,
        infergoal_dir=INFERGOAL_DIR,
        attn_dir=ATTENTION_DIR,
        out_root=OUT,
        pattern="*.npy"
    )
