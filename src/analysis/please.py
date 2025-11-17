from typing import Dict, Optional, Tuple
import numpy as np

def _regularize_cov(cov: np.ndarray, eps: float) -> np.ndarray:
    d = cov.shape[0]
    return cov + eps * np.eye(d, dtype=cov.dtype)

def _safe_logdet(evals: np.ndarray, eps: float) -> float:
    ev = np.clip(evals, eps, None)
    return float(np.sum(np.log(ev)))

def _spectral_entropy_and_erank(evals: np.ndarray, eps: float):
    s = float(np.sum(evals))
    if s <= eps:
        return 0.0, 1.0
    p = np.clip(evals / s, eps, None)
    p /= p.sum()
    vN = float(-np.sum(p * np.log(p)))
    er = float(np.exp(vN))
    return vN, er

def _condition_number(evals: np.ndarray, eps: float) -> float:
    ev = np.clip(evals, eps, None)
    return float(ev.max() / ev.min())

def temporal_uncertainty_metrics(
    X: np.ndarray,
    eps: float = 1e-6,
    use_gaussian_fisher: bool = True,
    pi_low: float = 0.05,
    pi_high: float = 0.95,
) -> Dict[str, np.ndarray]:
    X = np.asarray(X)
    if X.ndim == 4 and X.shape[1] == 1:
        S, _, T, D = X.shape
        X = X[:, 0, :, :]
    elif X.ndim == 3:
        S, T, D = X.shape
    else:
        raise ValueError("X must have shape (S,1,T,D) or (S,T,D).")

    mu = X.mean(axis=0)
    Y  = X - mu

    trace = Y.var(axis=0, ddof=1).sum(axis=-1)

    norms = np.linalg.norm(X, axis=-1)
    lo = np.quantile(norms, 0.05, axis=0)
    hi = np.quantile(norms, 0.95, axis=0)
    pi_width = hi - lo

    dmu = np.zeros(T, dtype=np.float64)
    dSigma = np.zeros(T, dtype=np.float64)

    logdet = np.zeros(T, dtype=np.float64)
    vN = np.zeros(T, dtype=np.float64)
    erank = np.zeros(T, dtype=np.float64)
    cond = np.zeros(T, dtype=np.float64)
    fisher_proxy = np.zeros(T, dtype=np.float64) if use_gaussian_fisher else None

    prev_cov = None
    for t in range(T):
        Yt = Y[:, t, :]
        cov = (Yt.T @ Yt) / max(S - 1, 1)
        cov = _regularize_cov(cov, 1e-6)

        evals = np.linalg.eigvalsh(cov)
        logdet[t] = _safe_logdet(evals, 1e-6)
        vN[t], erank[t] = _spectral_entropy_and_erank(evals, 1e-6)
        cond[t] = _condition_number(evals, 1e-6)
        if use_gaussian_fisher:
            fisher_proxy[t] = float(np.sum(1.0 / np.clip(evals, 1e-6, None)))

        if prev_cov is not None:
            dSigma[t] = np.linalg.norm(cov - prev_cov, ord='fro')
        prev_cov = cov
        if t > 0:
            dmu[t] = float(np.linalg.norm(mu[t] - mu[t-1]))

    metrics = dict(
        logdet=logdet,
        trace=trace,
        vN=vN,
        erank=erank,
        cond=cond,
        PI_norm_width=pi_width,
        dmu=dmu,
        dSigma=dSigma,
    )
    if use_gaussian_fisher:
        metrics["fisher_proxy"] = fisher_proxy
    return metrics

def temporal_uncertainty_U(
    X: np.ndarray,
    metric: str = "vN",
    g_schedule: Optional[np.ndarray] = None,
    eps: float = 1e-6,
):
    mts = temporal_uncertainty_metrics(X, eps=eps, use_gaussian_fisher=True)
    key_map = {"vN":"vN","logdet":"logdet","trace":"trace","fisher":"fisher_proxy","PI":"PI_norm_width"}
    if metric not in key_map:
        raise ValueError(f"metric must be one of {list(key_map.keys())}")
    m = mts[key_map[metric]]
    T = m.shape[0]
    if g_schedule is None:
        w = np.ones(T, dtype=np.float64)
    else:
        g_schedule = np.asarray(g_schedule, dtype=np.float64).reshape(-1)
        if g_schedule.shape[0] != T:
            raise ValueError("g_schedule must have length T")
        w = g_schedule ** 4
    U = w * m
    return U, mts