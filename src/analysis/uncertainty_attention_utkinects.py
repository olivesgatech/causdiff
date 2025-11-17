# # # import os, glob, re, json
# # # from typing import List, Dict, Tuple
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # # -----------------------------
# # # # 자연 정렬 & 경로 유틸
# # # # -----------------------------
# # # def _natural_key(s: str):
# # #     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# # # def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
# # #     paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
# # #     return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# # # def to_stem(p: str) -> str:
# # #     return os.path.splitext(os.path.basename(p))[0]

# # # # -----------------------------
# # # # 확률/엔트로피 유틸
# # # # -----------------------------
# # # def sanitize(arr: np.ndarray) -> np.ndarray:
# # #     arr = np.array(arr, dtype=np.float64, copy=True)
# # #     arr[~np.isfinite(arr)] = 0.0
# # #     return arr

# # # def looks_prob(arr: np.ndarray, tol=1e-3) -> bool:
# # #     if arr.ndim < 1: return False
# # #     row_sum = np.nansum(arr, axis=-1)
# # #     return np.all((row_sum > 1 - tol) & (row_sum < 1 + tol)) and np.nanmin(arr) >= -tol

# # # def softmax_last(x: np.ndarray) -> np.ndarray:
# # #     x = x - np.nanmax(x, axis=-1, keepdims=True)
# # #     ex = np.exp(x)
# # #     s = np.nansum(ex, axis=-1, keepdims=True)
# # #     s[s == 0] = 1.0
# # #     return ex / s

# # # def ensure_prob(arr: np.ndarray) -> np.ndarray:
# # #     return arr if looks_prob(arr) else softmax_last(arr)

# # # def entropy_over_last(p: np.ndarray, eps=1e-12) -> np.ndarray:
# # #     # 마지막 축 엔트로피
# # #     p = np.clip(p, eps, 1.0)
# # #     return -np.sum(p * np.log(p), axis=-1)

# # # # -----------------------------
# # # # 어텐션 로딩(공통)
# # # # -----------------------------
# # # def load_attention_1d(attn_path: str) -> np.ndarray:
# # #     a = np.load(attn_path, allow_pickle=False)
# # #     a = sanitize(a)
# # #     if a.ndim == 3 and a.shape[1] == 1:
# # #         a = a.mean(axis=0)[0]   # (T,)
# # #     elif a.ndim == 2:
# # #         a = a.mean(axis=0)      # (T,)
# # #     elif a.ndim == 1:
# # #         pass
# # #     else:
# # #         raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")
# # #     s = a.sum()
# # #     if s != 0: a = a / s
# # #     return a

# # # # -----------------------------
# # # # 불확실성 계산기
# # # # -----------------------------

# # # def time_unc_infergoal(stc_path: str) -> Tuple[np.ndarray, Dict]:
# # #     x = np.load(stc_path, allow_pickle=False)
# # #     if x.ndim == 4:
# # #         x = x[:,0,:,:]
# # #     elif x.ndim != 3:
# # #         raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C), got {x.shape}")
# # #     S,T,C = x.shape
# # #     probs = ensure_prob(sanitize(x))  # 피처/로짓이면 softmax
# # #     H_st = entropy_over_last(probs)
# # #     return H_st.mean(axis=0), {"S":S,"T":T,"C":C}

# # # # -----------------------------
# # # # 상관/커버리지 & 플롯
# # # # -----------------------------
# # # def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
# # #     m = min(a.size, b.size)
# # #     a, b = a[:m], b[:m]
# # #     if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
# # #     return float(np.corrcoef(a, b)[0,1])

# # # def attention_coverage_on_topq(attn: np.ndarray, unc: np.ndarray, q=0.1) -> float:
# # #     T = min(attn.size, unc.size)
# # #     attn = attn[:T]; unc = unc[:T]
# # #     k = max(1, int(round(T * q)))
# # #     idx = np.argpartition(-unc, k-1)[:k]
# # #     return float(attn[idx].sum())

# # # def plot_unc_vs_attn(unc_t: np.ndarray, attn_t: np.ndarray, title: str, out_path: str):
# # #     T = min(unc_t.size, attn_t.size)
# # #     x = np.arange(T)
# # #     fig, ax1 = plt.subplots(figsize=(12, 3.2))
# # #     ax1.plot(x, unc_t[:T], linewidth=2)
# # #     ax1.set_xlabel("Time"); ax1.set_ylabel("Entropy (H)", color="C0")
# # #     ax2 = ax1.twinx()
# # #     ax2.plot(x, attn_t[:T], linewidth=1.3, alpha=0.9)
# # #     ax2.set_ylabel("Attention (normalized)", color="C1")
# # #     ax1.set_title(title)
# # #     fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

# # # def run_with_shared_attention(
# # #     infergoal_dir: str,
# # #     attn_dir: str,
# # #     out_root: str,
# # #     pattern: str = "*.npy"
# # # ):
# # #     os.makedirs(out_root, exist_ok=True)
# # #     g_files = list_sorted_files(infergoal_dir, pattern)
# # #     a_files = list_sorted_files(attn_dir, pattern)
# # #     print(g_files, a_files)

# # #     if not g_files or not a_files:
# # #         raise RuntimeError("One or more folders are empty.")

# # #     # 인덱스 기준 최소 길이만큼만 비교
# # #     N = min(len(g_files), len(a_files))
# # #     if len({len(g_files), len(a_files)}) != 1:
# # #         print(f"[WARN] count mismatch: infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

# # #     out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

# # #     rows = []
# # #     print(f"Using shared attention from: {attn_dir}")
# # #     print(f"#pairs (by index) = {N}")

# # #     for i in range(N):
        
# # #         g_path = g_files[i]
# # #         a_path = a_files[i]
# # #         stem   = os.path.splitext(os.path.basename(g_path))[0]  # 로그/파일명 표기를 위해

# # #         # 공통 attention
# # #         attn_t = load_attention_1d(a_path)

# # #         # infer_goal uncertainty (그룹 매핑 없이)
# # #         unc_g, meta_g = time_unc_infergoal(g_path)
# # #         r_g = pearson_corr(unc_g, attn_t)
# # #         cov10_g = attention_coverage_on_topq(attn_t, unc_g, 0.10)
# # #         cov20_g = attention_coverage_on_topq(attn_t, unc_g, 0.20)
# # #         plot_unc_vs_attn(unc_g, attn_t,
# # #                          f"infer_goal: {stem}",
# # #                          os.path.join(out_g, f"{stem}_unc_vs_attn.png"))

# # #         rows.append({
# # #             "idx": i,
# # #             "infer_goal_file": os.path.basename(g_path),
# # #             "attention_file": os.path.basename(a_path),
# # #             "r_infer_goal": r_g, "cov10_infer_goal": cov10_g, "cov20_infer_goal": cov20_g,
# # #             "S_g": meta_g["S"], "T_g": meta_g["T"], "C_g": meta_g["C"],
# # #         })
# # #         print(f"[{i:03d}], r_g={r_g:.3f}")

# # #     df = pd.DataFrame(rows)
# # #     csv_path = os.path.join(out_root, "summary_shared_attention_by_index.csv")
# # #     df.to_csv(csv_path, index=False)

# # #     agg = {
# # #         "num_files": int(len(df)),
# # #         "infer_goal": {
# # #             "mean_r": float(df["r_infer_goal"].mean()),
# # #             "std_r": float(df["r_infer_goal"].std(ddof=1)) if len(df) > 1 else 0.0,
# # #             "mean_cov10": float(df["cov10_infer_goal"].mean()),
# # #             "mean_cov20": float(df["cov20_infer_goal"].mean()),
# # #         },
# # #         "summary_csv": csv_path,
# # #         "out_root": out_root
# # #     }
# # #     with open(os.path.join(out_root, "aggregate_shared_attention_by_index.json"), "w") as f:
# # #         json.dump(agg, f, indent=2)
# # #     print("\n=== Aggregate (shared attention; by index) ===")
# # #     print(json.dumps(agg, indent=2))


# # # # -----------------------------
# # # # main
# # # # -----------------------------
# # # if __name__ == "__main__":
# # #     BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects"
# # #     INFERGOAL_DIR = os.path.join(BASE, "infer_goal/30")
# # #     ATTENTION_DIR = os.path.join(BASE, "attention_map")
# # #     OUT = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects/uncertainty_attention_out_shared"

# # #     run_with_shared_attention(
# # #         infergoal_dir=INFERGOAL_DIR,
# # #         attn_dir=ATTENTION_DIR,
# # #         out_root=OUT,
# # #         pattern="*.npy"
# # #     )
# # import os, glob, re, json
# # from typing import List, Dict, Tuple
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # -----------------------------
# # # 자연 정렬 & 경로 유틸
# # # -----------------------------
# # def _natural_key(s: str):
# #     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# # def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
# #     paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
# #     return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# # def to_stem(p: str) -> str:
# #     return os.path.splitext(os.path.basename(p))[0]

# # # -----------------------------
# # # 확률/엔트로피 유틸
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

# # def entropy_over_last(p: np.ndarray, eps=1e-12) -> np.ndarray:
# #     # 마지막 축(C)에 대한 엔트로피
# #     p = np.clip(p, eps, 1.0)
# #     return -np.sum(p * np.log(p), axis=-1)

# # # -----------------------------
# # # 어텐션 로딩 (S,T) 형태 유지
# # # -----------------------------
# # def load_attention_ST(attn_path: str) -> np.ndarray:
# #     """
# #     반환: (S, T)
# #     허용 입력:
# #       - (S, 1, T)  → squeeze(1) → (S, T)
# #       - (S, T)     → 그대로
# #       - (T,)       → (1, T)로 승격
# #     각 샘플별 합이 1이 되도록 정규화.
# #     """
# #     a = np.load(attn_path, allow_pickle=False)
# #     a = sanitize(a)

# #     if a.ndim == 3 and a.shape[1] == 1:
# #         a = a[:, 0, :]           # (S, T)
# #     elif a.ndim == 2:
# #         pass                     # (S, T)
# #     elif a.ndim == 1:
# #         a = a[None, :]           # (1, T)
# #     else:
# #         raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")

# #     row_sum = a.sum(axis=1, keepdims=True)
# #     row_sum[row_sum == 0] = 1.0
# #     a = a / row_sum
# #     return a  # (S, T)

# # # -----------------------------
# # # 불확실성 계산기 (S,T) 반환 옵션
# # # -----------------------------
# # def time_unc_infergoal(stc_path: str, return_per_sample: bool = False) -> Tuple[np.ndarray, Dict]:
# #     """
# #     입력 stc_path: (S,T,C) 또는 (S,B,T,C). B가 있으면 B=0만 사용.
# #     return_per_sample:
# #       - False -> (T,) (S 평균)
# #       - True  -> (S,T)
# #     """
# #     x = np.load(stc_path, allow_pickle=False)
# #     if x.ndim == 4:
# #         x = x[:, 0, :, :]   # (S,T,C)
# #     elif x.ndim != 3:
# #         raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C) or (S,B,T,C), got {x.shape}")
# #     S, T, C = x.shape
# #     probs = ensure_prob(sanitize(x))  # (S,T,C)
# #     H_st = entropy_over_last(probs)   # (S,T)
# #     if return_per_sample:
# #         return H_st, {"S": S, "T": T, "C": C}
# #     else:
# #         return H_st.mean(axis=0), {"S": S, "T": T, "C": C}

# # # -----------------------------
# # # 오버레이 플로터 (하나의 파일에 2행: 위 H, 아래 A)
# # # -----------------------------
# # def plot_overlay_both(unc_ST: np.ndarray, attn_ST: np.ndarray, title: str, out_path: str):
# #     """
# #     unc_ST: (S,T)  - 엔트로피(불확실성)
# #     attn_ST: (S,T) - 어텐션(샘플별 정규화)
# #     두 메트릭을 같은 x축(T)에 대해 각각 겹쳐 그린 2행 서브플롯 그림 한 장 저장.
# #     """
# #     S_u, T_u = unc_ST.shape
# #     S_a, T_a = attn_ST.shape
# #     S = min(S_u, S_a)
# #     T = min(T_u, T_a)

# #     x = np.arange(T)
# #     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.5), sharex=True)

# #     # 1행: 불확실성
# #     ax1 = axes[0]
# #     for s in range(S):
# #         ax1.plot(x, unc_ST[s, :T], linewidth=1.6, alpha=0.9, label=f"S{s}")
# #     ax1.set_ylabel("Entropy (H)")
# #     ax1.set_title(title + " — Uncertainty Overlay")
# #     ax1.grid(True, alpha=0.3)
# #     ax1.legend(loc="upper right", ncol=min(S,5))

# #     # 2행: 어텐션
# #     ax2 = axes[1]
# #     for s in range(S):
# #         ax2.plot(x, attn_ST[s, :T], linewidth=1.3, alpha=0.9, label=f"S{s}")
# #     ax2.set_xlabel("Time")
# #     ax2.set_ylabel("Attention (normalized)")
# #     ax2.set_title("Attention Overlay")
# #     ax2.grid(True, alpha=0.3)
# #     ax2.legend(loc="upper right", ncol=min(S,5))

# #     fig.tight_layout()
# #     fig.savefig(out_path, dpi=200)
# #     plt.close(fig)

# # # -----------------------------
# # # 실행 루틴
# # # -----------------------------
# # def run_with_shared_attention(
# #     infergoal_dir: str,
# #     attn_dir: str,
# #     out_root: str,
# #     pattern: str = "*.npy"
# # ):
# #     os.makedirs(out_root, exist_ok=True)
# #     g_files = list_sorted_files(infergoal_dir, pattern)
# #     a_files = list_sorted_files(attn_dir, pattern)

# #     if not g_files or not a_files:
# #         raise RuntimeError("One or more folders are empty.")

# #     # 파일 수가 다르면 공통 최소치만 사용
# #     N = min(len(g_files), len(a_files))
# #     if len({len(g_files), len(a_files)}) != 1:
# #         print(f"[WARN] count mismatch: infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

# #     out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

# #     print(f"Using shared attention from: {attn_dir}")
# #     print(f"#pairs (by index) = {N}")

# #     for i in range(N):
# #         g_path = g_files[i]
# #         a_path = a_files[i]
# #         stem   = os.path.splitext(os.path.basename(g_path))[0]

# #         # (S,T) 로드
# #         attn_ST = load_attention_ST(a_path)
# #         unc_ST, meta_g = time_unc_infergoal(g_path, return_per_sample=True)  # (S,T)

# #         # 오버레이 한 장 저장
# #         out_png = os.path.join(out_g, f"{stem}_overlay.png")
# #         plot_overlay_both(
# #             unc_ST=unc_ST,
# #             attn_ST=attn_ST,
# #             title=f"infer_goal: {stem}",
# #             out_path=out_png
# #         )
# #         print(f"[{i:03d}] saved overlay -> {out_png}")

# # # -----------------------------
# # # main
# # # -----------------------------
# # if __name__ == "__main__":
# #     BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects"
# #     INFERGOAL_DIR = os.path.join(BASE, "infer_goal/30")
# #     ATTENTION_DIR = os.path.join(BASE, "attention_map")
# #     OUT = os.path.join(BASE, "uncertainty_attention_out_shared")

# #     run_with_shared_attention(
# #         infergoal_dir=INFERGOAL_DIR,
# #         attn_dir=ATTENTION_DIR,
# #         out_root=OUT,
# #         pattern="*.npy"
# #     )
# import os, glob, re, json
# from typing import List, Tuple
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # 자연 정렬 & 경로 유틸
# # -----------------------------
# def _natural_key(s: str):
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# def list_sorted_files(folder: str, pattern: str = "*.npy") -> List[str]:
#     paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
#     return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

# # -----------------------------
# # 수치/확률/엔트로피 유틸
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
#     return arr if looks_prob(arr) else softmax_last(arr)

# def entropy_over_last(p: np.ndarray, eps=1e-12) -> np.ndarray:
#     # 마지막 축(C)에 대한 엔트로피
#     p = np.clip(p, eps, 1.0)
#     return -np.sum(p * np.log(p), axis=-1)

# def minmax_normalize_1d(x: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, dtype=np.float64)
#     xmin, xmax = np.min(x), np.max(x)
#     if xmax - xmin == 0:
#         return np.zeros_like(x)
#     y = (x - xmin) / (xmax - xmin)
#     y[~np.isfinite(y)] = 0.0
#     return y

# # -----------------------------
# # 어텐션 로딩 (S,T) 형태 유지
# # -----------------------------
# def load_attention_ST(attn_path: str) -> np.ndarray:
#     """
#     반환: (S, T)
#     허용 입력:
#       - (S, 1, T)  → squeeze(1) → (S, T)
#       - (S, T)     → 그대로
#       - (T,)       → (1, T)로 승격
#     각 샘플별 합이 1이 되도록 정규화.
#     """
#     a = np.load(attn_path, allow_pickle=False)
#     a = sanitize(a)

#     if a.ndim == 3 and a.shape[1] == 1:
#         a = a[:, 0, :]           # (S, T)
#     elif a.ndim == 2:
#         pass                     # (S, T)
#     elif a.ndim == 1:
#         a = a[None, :]           # (1, T)
#     else:
#         raise ValueError(f"Unsupported attention shape {a.shape} in {os.path.basename(attn_path)}")

#     row_sum = a.sum(axis=1, keepdims=True)
#     row_sum[row_sum == 0] = 1.0
#     a = a / row_sum
#     return a  # (S, T)

# # -----------------------------
# # 불확실성: S축을 통합해 시간별 하나의 엔트로피 (T,)
# # -----------------------------
# def time_unc_from_samples(stc_path: str) -> Tuple[np.ndarray, dict]:
#     """
#     입력 stc_path: (S,T,C) 또는 (S,B,T,C). B가 있으면 B=0만 사용.
#     처리:
#       1) 확률화 (클래스 축 C)
#       2) S축 평균: p_bar(t,c) = mean_s p_s(t,c)  → (T,C)
#       3) 엔트로피 H_t = -sum_c p_bar(t,c) log p_bar(t,c) → (T,)
#     반환: (H_t, meta)
#     """
#     x = np.load(stc_path, allow_pickle=False)
#     if x.ndim == 4:
#         x = x[:, 0, :, :]   # (S,T,C)
#     elif x.ndim != 3:
#         raise ValueError(f"{os.path.basename(stc_path)} expected (S,T,C) or (S,B,T,C), got {x.shape}")
#     S, T, C = x.shape
#     probs = ensure_prob(sanitize(x))      # (S,T,C)
#     p_bar = probs.mean(axis=0)            # (T,C)
#     H_t = entropy_over_last(p_bar)        # (T,)
#     return H_t, {"S": S, "T": T, "C": C}

# # -----------------------------
# # 상관/플로팅
# # -----------------------------
# def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
#     m = min(a.size, b.size)
#     a, b = a[:m], b[:m]
#     if np.std(a) < 1e-12 or np.std(b) < 1e-12:
#         return 0.0
#     return float(np.corrcoef(a, b)[0, 1])

# def plot_overlay_entropy_and_attn(Hn_t: np.ndarray, attn_ST: np.ndarray, title: str, out_path: str):
#     """
#     Hn_t: (T,)      - 정규화된 엔트로피(단일)
#     attn_ST: (S,T)  - 샘플별 정규화된 어텐션
#     2행 서브플롯: 위 = Hn_t 하나, 아래 = 어텐션 S개 오버레이
#     """
#     T_use = min(Hn_t.size, attn_ST.shape[1])
#     x = np.arange(T_use)

#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.0), sharex=True)

#     # 1행: 정규화된 불확실성 (단일 곡선)
#     ax1 = axes[0]
#     ax1.plot(x, Hn_t[:T_use], linewidth=2.0)
#     ax1.set_ylabel("Normalized Entropy")
#     ax1.set_title(title + " — Uncertainty (single)")
#     ax1.grid(True, alpha=0.3)

#     # 2행: 어텐션 (S개 오버레이)
#     ax2 = axes[1]
#     S = attn_ST.shape[0]
#     for s in range(S):
#         ax2.plot(x, attn_ST[s, :T_use], linewidth=1.2, alpha=0.9, label=f"S{s}")
#     ax2.set_xlabel("Time")
#     ax2.set_ylabel("Attention (normalized)")
#     ax2.set_title("Attention Overlay")
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(loc="upper right", ncol=min(S, 5))

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)

# # -----------------------------
# # 실행 루틴
# # -----------------------------
# def run_with_shared_attention(
#     infergoal_dir: str,
#     attn_dir: str,
#     out_root: str,
#     pattern: str = "*.npy"
# ):
#     os.makedirs(out_root, exist_ok=True)
#     g_files = list_sorted_files(infergoal_dir, pattern)
#     a_files = list_sorted_files(attn_dir, pattern)

#     if not g_files or not a_files:
#         raise RuntimeError("One or more folders are empty.")

#     # 파일 수가 다르면 공통 최소치만 사용
#     N = min(len(g_files), len(a_files))
#     if len({len(g_files), len(a_files)}) != 1:
#         print(f"[WARN] count mismatch: infer_goal={len(g_files)}, attention={len(a_files)}. Using N={N} by index.")

#     out_g = os.path.join(out_root, "infer_goal"); os.makedirs(out_g, exist_ok=True)

#     print(f"Using shared attention from: {attn_dir}")
#     print(f"#pairs (by index) = {N}")

#     # 집계 버킷
#     per_file_max_r = []   # 각 파일에서 max_s r
#     all_sample_r = []     # 모든 파일×샘플 r
#     rows = []             # CSV 요약

#     for i in range(N):
#         g_path = g_files[i]
#         a_path = a_files[i]
#         stem   = os.path.splitext(os.path.basename(g_path))[0]

#         # (S,T) 로드
#         attn_ST = load_attention_ST(a_path)          # (S_a, T_a)

#         # 불확실성: S축 통합 → (T,)
#         H_t, meta = time_unc_from_samples(g_path)    # (T,)
#         Hn_t = minmax_normalize_1d(H_t)              # (T,) 정규화

#         # 공통 길이
#         T_use = min(Hn_t.size, attn_ST.shape[1])
#         S_use = attn_ST.shape[0]
#         attn_use = attn_ST[:, :T_use]
#         Hn_use   = Hn_t[:T_use]

#         # 플롯 (엔트로피 단일 + 어텐션 오버레이)
#         out_png = os.path.join(out_g, f"{stem}_overlay_normH_single.png")
#         plot_overlay_entropy_and_attn(
#             Hn_t=Hn_use,
#             attn_ST=attn_use,
#             title=f"infer_goal: {stem}",
#             out_path=out_png
#         )
#         print(f"[{i:03d}] saved overlay -> {out_png}")

#         # 샘플별 상관계수: corr(attn[s], Hn_t)
#         r_list = []
#         for s in range(S_use):
#             r = pearson_corr(attn_use[s], Hn_use)
#             r_list.append(r)
#             rows.append({
#                 "idx": i,
#                 "sample": s,
#                 "infer_goal_file": os.path.basename(g_path),
#                 "attention_file": os.path.basename(a_path),
#                 "r_corr": r,
#                 "S": meta["S"], "T": meta["T"], "C": meta["C"],
#                 "T_use": T_use,
#             })

#         if r_list:
#             per_file_max_r.append(float(np.min(r_list)))
#             all_sample_r.extend(r_list)

#     # 통계 요약
#     max_r_mean = float(np.mean(per_file_max_r)) if per_file_max_r else 0.0
#     overall_r_mean = float(np.mean(all_sample_r)) if all_sample_r else 0.0

#     # CSV/JSON 저장
#     df = pd.DataFrame(rows)
#     csv_path = os.path.join(out_root, "summary_shared_attention_by_index.csv")
#     df.to_csv(csv_path, index=False)

#     agg = {
#         "num_files": int(len(per_file_max_r)),
#         "num_rows": int(len(rows)),
#         "mean_of_max_r_per_file": max_r_mean,   # 각 파일의 max 상관계수들의 평균
#         "overall_mean_r": overall_r_mean,       # 모든 샘플 상관계수의 단순 평균
#         "summary_csv": csv_path,
#         "out_root": out_root
#     }
#     with open(os.path.join(out_root, "aggregate_shared_attention_by_index.json"), "w") as f:
#         json.dump(agg, f, indent=2)

#     print("\n=== Aggregate (shared attention; by index) ===")
#     print(json.dumps(agg, indent=2))

# # -----------------------------
# # main
# # -----------------------------
# if __name__ == "__main__":
#     BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects"
#     INFERGOAL_DIR = os.path.join(BASE, "infer_goal/30")
#     ATTENTION_DIR = os.path.join(BASE, "attention_map")
#     OUT = os.path.join(BASE, "uncertainty_attention_out_shared")

#     run_with_shared_attention(
#         infergoal_dir=INFERGOAL_DIR,
#         attn_dir=ATTENTION_DIR,
#         out_root=OUT,
#         pattern="*.npy"
#     )
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

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    #BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/utkinects"
    #BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/darai_l3"
    BASE = "/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd"
    INFERGOAL_DIR = os.path.join(BASE, "infer_goal/30")
    ATTENTION_DIR = os.path.join(BASE, "attention_map_30")
    OUT = os.path.join(BASE, "uncertainty_attention_out_shared_30")

    run_with_shared_attention(
        infergoal_dir=INFERGOAL_DIR,
        attn_dir=ATTENTION_DIR,
        out_root=OUT,
        pattern="*.npy"
    )
