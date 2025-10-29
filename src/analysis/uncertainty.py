import numpy as np
from typing import Optional, Sequence, Dict, Any

def _to_numpy(x):
    try:
        import torch  # optional
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _safe_norm(x, axis=-1, eps=1e-12):
    return np.sqrt(np.maximum((x**2).sum(axis=axis, keepdims=True), eps))

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / _safe_norm(a, axis=-1)
    b_norm = b / _safe_norm(b, axis=-1)
    return np.tensordot(a_norm, b_norm.T, axes=([a.ndim-1],[0]))

def _softmax(z: np.ndarray, axis: int = -1, tau: float = 1.0) -> np.ndarray:
    z = z / max(tau, 1e-12)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)

def _entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)

def _stable_rank_from_cov(cov: np.ndarray, eps: float = 1e-12) -> float:
    # stable rank = (trace(AA^T)) / sigma_max^2, but here use Frobenius^2 / spectral^2
    # Cheaper proxy: (trace(cov))^2 / trace(cov @ cov)  (<= effective rank; good surrogate)
    tr = np.trace(cov)
    tr2 = np.sum(cov*cov)
    if tr2 <= eps:
        return 0.0
    return float((tr*tr) / max(tr2, eps))

def compute_probabilities_from_class_embeddings(
    samples: np.ndarray,          # (S, T, D)
    class_embs: np.ndarray,       # (C, D)
    tau: float = 1.0,
    use_cosine: bool = True,
) -> np.ndarray:
    S, T, D = samples.shape
    if use_cosine:
        logits = _cosine_sim(samples.reshape(S*T, D), class_embs)  # (S*T, C)
    else:
        logits = samples.reshape(S*T, D) @ class_embs.T            # (S*T, C)
    p = _softmax(logits, axis=-1, tau=tau).reshape(S, T, -1)
    return p

def compute_semantic_uncertainty_metrics(
    probs: np.ndarray,                 # (S, T, C)
    goal_emb: Optional[np.ndarray] = None,     # (D,) in same space as class_embs
    class_embs: Optional[np.ndarray] = None,   # (C, D)
    allowed_classes: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    S, T, C = probs.shape
    pbar = probs.mean(axis=0)                 # (T, C)
    ent_pred = _entropy(pbar, axis=-1)        # (T,)
    ent_per_sample = _entropy(probs, axis=-1) # (S, T)
    ent_ale = ent_per_sample.mean(axis=0)     # (T,)
    ent_epi = ent_pred - ent_ale              # (T,)

    # agreement & margin
    top1 = probs.argmax(axis=-1)              # (S, T)
    agree = np.zeros((T,), dtype=float)
    for t in range(T):
        vals, counts = np.unique(top1[:, t], return_counts=True)
        agree[t] = counts.max() / float(S)
    sorted_p = np.sort(pbar, axis=-1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]

    out_mass = None
    in_entropy = None
    gc_bald_epi = None
    sdi = None

    if allowed_classes is not None:
        mask = np.zeros((C,), dtype=bool)
        mask[np.asarray(allowed_classes, dtype=int)] = True
        M_in = (pbar[:, mask]).sum(axis=-1)              # (T,)
        M_out = (pbar[:, ~mask]).sum(axis=-1)            # (T,)
        out_mass = M_out

        in_norm = np.divide(pbar[:, mask], np.clip(M_in[:, None], 1e-12, None))
        in_entropy = _entropy(in_norm, axis=-1)

        ps_in = np.divide(probs[:, :, mask], np.clip(probs[:, :, mask].sum(axis=-1, keepdims=True), 1e-12, None))
        ent_in_pred = _entropy(in_norm, axis=-1)
        ent_in_per_sample = _entropy(ps_in, axis=-1).mean(axis=0)
        gc_bald_epi = ent_in_pred - ent_in_per_sample

    if (goal_emb is not None) and (class_embs is not None):
        g = goal_emb / np.clip(np.linalg.norm(goal_emb), 1e-12, None)
        ce = class_embs / np.clip(np.linalg.norm(class_embs, axis=-1, keepdims=True), 1e-12, None)
        cos = (ce @ g.reshape(-1))  # (C,)
        dist = 1.0 - cos
        sdi = (pbar * dist[None, :]).sum(axis=-1)  # (T,)

    return {
        "pred_entropy": ent_pred,
        "aleatoric_entropy": ent_ale,
        "epistemic_entropy": ent_epi,
        "agreement": agree,
        "margin": margin,
        "out_of_goal_mass": out_mass,            # Optional
        "in_goal_entropy": in_entropy,           # Optional
        "gc_bald_epistemic": gc_bald_epi,        # Optional
        "semantic_deviation_index": sdi          # Optional
    }

def compute_temporal_structure_metrics(
    samples: np.ndarray,   # (S, T, D)
    window: int = 0
) -> Dict[str, np.ndarray]:
    S, T, D = samples.shape
    m_t = samples.mean(axis=0)  # (T, D)

    # Stable-rank proxy instead of full eigendecomposition (fast)
    stable_rank_S = np.zeros((T,), dtype=float)
    for t in range(T):
        X = samples[:, t, :]  # (S, D)
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / max(S - 1, 1)
        stable_rank_S[t] = _stable_rank_from_cov(cov)

    delta = np.linalg.norm(np.diff(m_t, axis=0), axis=-1)            # (T-1,)
    accel = np.linalg.norm(np.diff(m_t, n=2, axis=0), axis=-1)       # (T-2,)
    delta_l2 = np.zeros((T,), dtype=float); delta_l2[1:] = delta
    accel_l2 = np.zeros((T,), dtype=float); accel_l2[2:] = accel

    stable_rank_win = None
    if window and window > 1:
        stable_rank_win = np.zeros((T,), dtype=float)
        half = window // 2
        for t in range(T):
            t0 = max(0, t - half)
            t1 = min(T, t + half + 1)
            X = samples[:, t0:t1, :].reshape(-1, samples.shape[-1])
            Xc = X - X.mean(axis=0, keepdims=True)
            cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
            stable_rank_win[t] = _stable_rank_from_cov(cov)

    return {
        "stable_rank_across_samples": stable_rank_S,
        "delta_l2_mean_feature": delta_l2,
        "accel_l2_mean_feature": accel_l2,
        "stable_rank_windowed": stable_rank_win  # Optional
    }

def measure_uncertainty(
    samples_STD,                        # (S, T, D) numpy or torch
    class_embs: Optional[np.ndarray] = None,  # (C, D)
    goal_emb: Optional[np.ndarray] = None,    # (D,)
    allowed_classes: Optional[Sequence[int]] = None,
    tau: float = 1.0,
    use_cosine: bool = True,
    window:int = 0
):
    samples = _to_numpy(samples_STD).astype(np.float32)  # (S, T, D)
    S, T, D = samples.shape

    sem_metrics = {}
    if class_embs is not None:
        class_embs_np = _to_numpy(class_embs).astype(np.float32)
        probs = compute_probabilities_from_class_embeddings(samples, class_embs_np, tau=tau, use_cosine=use_cosine)
        sem_metrics = compute_semantic_uncertainty_metrics(
            probs=probs,
            goal_emb=None if goal_emb is None else _to_numpy(goal_emb).astype(np.float32),
            class_embs=class_embs_np,
            allowed_classes=allowed_classes
        )

    temp_metrics = compute_temporal_structure_metrics(samples, window=window)

    return {
        "semantic": sem_metrics,
        "temporal": temp_metrics
    }

# --- Minimal demo ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S, T, D, C = rng.shape
    base = rng.normal(size=(T, D)).astype(np.float32)
    base = np.cumsum(base * 0.02, axis=0)
    samples = base[None, :, :] + rng.normal(scale=0.15, size=(S, T, D)).astype(np.float32)
    class_embs = rng.normal(size=(C, D)).astype(np.float32)
    goal_emb = class_embs[0].copy()

    res = measure_uncertainty(
        samples,
        class_embs=class_embs,
        goal_emb=goal_emb,
        allowed_classes=[0,1,2],
        tau=0.7,
        window=7
    )
    # quick sanity print
    print({k: (None if v is None else np.asarray(list(v.values())[0]).shape) for k,v in res.items()})
