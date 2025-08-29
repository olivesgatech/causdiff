import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

def _spectrum_from_matrix(X: np.ndarray, center: bool = True) -> np.ndarray:
    """
    Compute normalized eigenvalue spectrum p from a matrix X of shape (N, D).
    Steps:
      - Center columns if center=True
      - SVD: Xc = U S V^T
      - eigenvalues of covariance ~ S^2 / (N-1)
      - Normalize to sum=1 to get p
    Returns p (length = min(N,D))
    """
    N, D = X.shape
    if N < 2 or D < 2:
        return np.array([])

    Xc = X - X.mean(axis=0, keepdims=True) if center else X.copy()
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (S ** 2) / max(N - 1, 1)
    total = lam.sum()
    if total <= 0 or not np.isfinite(total):
        return np.array([])
    return lam / total

def _effective_rank(p: np.ndarray, eps: float = 1e-12) -> float:
    if p.size == 0:
        return float("nan")
    p_safe = np.clip(p, eps, 1.0)
    H = -np.sum(p_safe * np.log(p_safe))
    return float(np.exp(H))

def analyze_folder_spectra(
    folder_path: str,
    save_prefix: str = "/mnt/data/eigenspectrum",
    center: bool = True,
    save_csv: bool = True
):
    """
    Analyze all .npy files in a folder with shape (S,1,D,T).
    Computes mean±std eigenvalue spectrum and mean effective rank.
    """
    npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {folder_path}")

    spectra = []
    eranks = []
    per_sample_records = []
    total_samples = 0

    for fpath in npy_files:
        arr = np.load(fpath, allow_pickle=False)
        if arr.ndim != 4 or arr.shape[1] != 1:
            print(f"Skipping {fpath}: expected shape (S,1,D,T), got {arr.shape}")
            continue

        S, _, D, T = arr.shape
        for s in range(S):
            block = arr[s, 0]  # shape (D, T)
            X = block.T        # shape (T, D) so that rows=time, cols=features
            p = _spectrum_from_matrix(X, center=center)
            if p.size == 0:
                continue
            spectra.append(p)
            er = _effective_rank(p)
            eranks.append(er)
            per_sample_records.append((os.path.basename(fpath), s, er))
            total_samples += 1

    if not spectra:
        raise RuntimeError("No valid spectra computed.")

    max_len = max(len(p) for p in spectra)
    padded = np.full((len(spectra), max_len), np.nan)
    for i, spec in enumerate(spectra):
        padded[i, :len(spec)] = spec

    mean_p = np.nanmean(padded, axis=0)
    std_p  = np.nanstd(padded, axis=0)
    er_arr = np.array(eranks)
    mean_erank = np.nanmean(er_arr)
    std_erank  = np.nanstd(er_arr)

    # Plot
    fig, ax = plt.subplots(figsize=(7,5))
    x = np.arange(1, max_len+1)
    ax.plot(x, mean_p, label="Mean normalized eigenvalue")
    ax.fill_between(x, mean_p - std_p, mean_p + std_p, alpha=0.3, label="±1 std")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Normalized eigenvalue")
    ax.set_title("Eigenvalue spectrum (S,1,D,T)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    plot_path = f"{save_prefix}_mean_std.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    summary_path = f"{save_prefix}_summary.txt"
    with open(summary_path, "w") as fout:
        fout.write(f"Analyzed folder: {folder_path}\n")
        fout.write(f"Num files: {len(npy_files)}\n")
        fout.write(f"Total valid samples: {total_samples}\n")
        fout.write(f"Mean effective rank: {mean_erank:.6f}\n")
        fout.write(f"Std effective rank:  {std_erank:.6f}\n")
        fout.write(f"Max spectrum length: {max_len}\n")

    # Optional CSV
    if save_csv:
        csv_path = f"{save_prefix}_per_sample_erank.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "sample_index", "effective_rank"])
            writer.writerows(per_sample_records)

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    if save_csv:
        print(f"Saved per-sample ERank CSV: {csv_path}")

    return mean_p, std_p, mean_erank, std_erank



# Point to your folder with ~34 .npy files
folder = "/mnt/data-tmp/seulgi/causdiff/src/output_baseline_48"

# Choose where to save outputs (prefix only; files will get suffixes)
save_prefix = "/mnt/data-tmp/seulgi/causdiff/src/eigenspectrum/baseline_48"

mean_p, std_p, mean_erank, std_erank = analyze_folder_spectra(
    folder_path=folder,
    save_prefix=save_prefix,
    center=True,     # center columns before SVD (recommended)
    save_csv=True    # also save per-sample effective ranks
)
