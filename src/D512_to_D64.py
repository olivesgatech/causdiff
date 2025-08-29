import os
import numpy as np

def reduce_file_S1T512_to_S1T64(input_path: str, output_folder: str, k: int = 64):
    """
    Load (S,1,T,512) array from input_path, fit PCA across all (S*T,512),
    reduce to (S,1,T,k), and save .npy file in output_folder.

    Args:
        input_path (str): path to input .npy file (shape (S,1,T,512))
        output_folder (str): directory to save reduced file
        k (int): number of PCA components (default 64)
    """
    # Load
    arr = np.load(input_path)
    if arr.ndim != 4 or arr.shape[1] != 1 or arr.shape[-1] != 512:
        raise ValueError(f"Expected (S,1,T,512), got {arr.shape}")
    S, _, T, D = arr.shape
    if k > D:
        raise ValueError(f"k={k} cannot exceed original dim {D}")

    # Reshape to (S*T, 512)
    X = arr.reshape(S*T, D)
    Xc = X - X.mean(axis=0)

    # Fit PCA with SVD
    U, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].T   # (512,k)
    reduced = (Xc @ components).reshape(S,1,T,k)  # (S,1,T,k)
    reduced = reduced.reshape(S,1,k,T)

    # Save
    os.makedirs(output_folder, exist_ok=True)
    base = os.path.basename(input_path)
    out_path = os.path.join(output_folder, base.replace(".npy", f"_pca{k}.npy"))
    np.save(out_path, reduced)
    print(f"Saved reduced array {reduced.shape} to {out_path}")

    return reduced, components

for file in os.listdir("/mnt/data-tmp/seulgi/causdiff/src/output_proposed"):
    if file.endswith(".npy"):
        input_path = os.path.join("/mnt/data-tmp/seulgi/causdiff/src/output_proposed", file)
        reduce_file_S1T512_to_S1T64(
            input_path=input_path,
            output_folder="/mnt/data-tmp/seulgi/causdiff/src/output_proposed_64",
            k=64
        )
