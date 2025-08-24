import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------
# Helpers
# ---------------------------------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _normalize_rows(x, eps=1e-9):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _effective_rank_from_eigvals(lam, eps=1e-12):
    lam = np.clip(np.asarray(lam), 0.0, None)
    s = lam.sum()
    if s <= eps:
        return 0.0
    p = lam / s
    H = -(p * np.log(p + eps)).sum()
    return float(np.exp(H))

# ---------------------------------------
# 1) Goal embeddings clustering & viz
# ---------------------------------------

def cluster_goal_embeddings(
    goal_embeddings,
    outfile,
    outdir='./src/visualize',
    n_clusters=None,
    method="umap",         # "umap" or "pca"
    metric="cosine",
    random_state=0,
    point_size=18,
    dpi=200
):
    """
    Args:
      goal_embeddings: np.ndarray [T, D] 혹은 [T*D](flat)
      outdir: 결과 저장 디렉토리
      n_clusters: 지정하지 않으면 silhouette로 2~10 사이에서 선택 (T<10이면 자동으로 min(T,10))
      method: "umap" 또는 "pca"
      metric: "cosine" (권장) 또는 "euclidean"
      random_state: 시드
      point_size: 시각화 점 크기
      dpi: 저장 DPI

    Saves:
      - clusters.csv: time,cluster
      - goal_2d_scatter.png: 2D 임베딩 산점도 (클러스터 색상)
      - goal_2d_scatter_nolabel.png: 축/범례 없는 버전
    Returns:
      dict(paths, labels, coords2d)
    """
    _ensure_dir(outdir)

    X = np.asarray(goal_embeddings)
    if X.ndim == 1:
        D = 512
        assert X.size % D == 0, f"flat 길이 {X.size}가 {D}로 나누어 떨어지지 않습니다."
        X = X.reshape(-1, D)
    assert X.ndim == 2, f"입력은 [T, D]여야 합니다. got {X.shape}"
    T, D = X.shape

    # 정규화 + 거리 메트릭에 맞는 전처리
    if metric == "cosine":
        Xn = _normalize_rows(X)
        X_for_cluster = Xn
    else:
        X_for_cluster = X.astype(np.float32)

    # 2D 투영
    if method.lower() == "umap":
        try:
            from umap import UMAP
            um = UMAP(n_components=2, n_neighbors=min(10, max(2, T//5)),
                      min_dist=0.1, metric=metric, random_state=random_state)
            Z2 = um.fit_transform(X_for_cluster)
        except Exception:
            # UMAP 미설치/실패 시 PCA fallback
            from sklearn.decomposition import PCA
            Z2 = PCA(n_components=2, random_state=random_state).fit_transform(X_for_cluster)
    else:
        from sklearn.decomposition import PCA
        Z2 = PCA(n_components=2, random_state=random_state).fit_transform(X_for_cluster)

    if n_clusters is None:
        ks = list(range(2, max(3, min(10, T)) + 1))
        best_k, best_score = None, -1.0
        for k in ks:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labs = km.fit_predict(X_for_cluster)
            # 샘플 수가 적거나 k가 크면 실루엣 계산 에러 방지
            try:
                sc = silhouette_score(X_for_cluster, labs, metric=("cosine" if metric=="cosine" else "euclidean"))
            except Exception:
                sc = -1.0
            if sc > best_score:
                best_k, best_score = k, sc
        n_clusters = best_k if best_k is not None else min(3, T)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_for_cluster)

    # 저장: 클러스터 CSV
    csv_path = os.path.join(outdir, f"{outfile}_clusters.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time,cluster\n")
        for t, c in enumerate(labels):
            f.write(f"{t},{int(c)}\n")

    # 시각화: 산점도
    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(i % 20) for i in labels])

    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=colors, s=point_size, edgecolors="none")
    plt.title(f"Goal embeddings clustering (k={n_clusters})")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.tight_layout()
    scatter_path = os.path.join(outdir, f"{outfile}_goal_2d_scatter.png")
    plt.savefig(scatter_path, dpi=dpi)
    plt.close()

    # 텍스트/축 없는 버전
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=colors, s=point_size, edgecolors="none")
    plt.axis("off")
    plt.tight_layout(pad=0)
    scatter_nolabel_path = os.path.join(outdir, f"{outfile}_goal_2d_scatter_nolabel.png")
    plt.savefig(scatter_nolabel_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def action_erank_and_spectrum(
    action_embeddings,
    outfile,
    outdir='./src/visualize_baseline',
    center=True,         # 전체 (N*T) 샘플 평균을 빼서 중심화
    row_normalize=True,  # 각 샘플(행)을 L2 정규화
    dpi=200
):
    """
    Args:
      action_embeddings: np.ndarray [N, 1, T, D] 또는 [N, T, D]
      outfile: 출력 파일 prefix (확장자 제외)
      outdir: 결과 저장 디렉토리
      center: 전체 샘플 평균 제거 후 Gram 계산
      row_normalize: 각 행 벡터를 단위노름으로 정규화 (스케일 영향 제거)

    Saves:
      - {outfile}_eigvals.npy        : 전체 스펙트럼 고유값 (내림차순)
      - {outfile}_spectrum.png       : 전체 스펙트럼 플롯 (log y)
      - {outfile}_erank.txt          : 단일 ERank 값
      - {outfile}_cum_energy.png     : 누적 에너지 곡선 (선택적 해석에 유용)

    Returns:
      dict(paths, values)
        - 'eigvals': np.ndarray [D]
        - 'erank'  : float
    """
    _ensure_dir(outdir)
    A = np.asarray(action_embeddings)
    if A.ndim == 4 and A.shape[1] == 1:
        A = A[:, 0, :, :]  # [N, T, D]
    assert A.ndim == 3, f"입력은 [N, T, D] 또는 [N, 1, T, D]여야 합니다. got {A.shape}"

    N, T, D = A.shape

    # (N*T, D)로 펴서 한 번에 분석
    X = A.reshape(N * T, D).astype(np.float64)  # [N*T, D]

    # 중심화 / 정규화
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if row_normalize:
        X = _normalize_rows(X)

    # Gram (D x D) — 스케일은 ERank에 영향 거의 없음
    G = X.T @ X  # [D, D]

    # 고유값 (대칭 행렬)
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0.0, None)[::-1]  # 내림차순 정렬

    # Effective Rank
    er = _effective_rank_from_eigvals(lam)

    # 저장 경로
    sp_png  = os.path.join(outdir, f"{outfile}_spectrum.png")
    er_txt  = os.path.join(outdir, f"{outfile}_erank.txt")
    cum_png = os.path.join(outdir, f"{outfile}_cum_energy.png")

    # 저장: 고유값 / ERank
    with open(er_txt, "w") as f:
        f.write(f"ERank: {er:.6f}\n")
        f.write(f"N: {N}, T: {T}, D: {D}\n")
        f.write(f"Sum eigenvalues: {lam.sum():.6f}\n")

    # 스펙트럼 플롯
    plt.figure(figsize=(6, 4))
    plt.plot(lam, linewidth=2)
    plt.yscale("log")
    plt.xlabel("eigen-index")
    plt.ylabel("eigenvalue (log)")
    plt.title("Eigenvalue spectrum (all times aggregated)")
    plt.tight_layout()
    plt.savefig(sp_png, dpi=dpi)
    plt.close()

    # 누적 에너지(정규화) 플롯: 상위 k 성분이 차지하는 비율
    total = lam.sum() + 1e-12
    cum = np.cumsum(lam) / total
    plt.figure(figsize=(6, 4))
    plt.plot(cum, linewidth=2)
    plt.xlabel("eigen-index (top-k)")
    plt.ylabel("cumulative energy")
    plt.title("Cumulative energy of spectrum")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(cum_png, dpi=dpi)
    plt.close()