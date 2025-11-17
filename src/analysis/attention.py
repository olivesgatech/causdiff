# save as: plot_attention_folder.py
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def plot_cs_attn_heatmap_array(w_2d: np.ndarray, save_path: str, title="CS attention"):
    """
    w_2d: (H, T) 형태의 ndarray. (여기서는 H=1)
    """
    if w_2d.ndim != 2:
        raise ValueError(f"Expected 2D array (H,T), got shape {w_2d.shape}")
    plt.figure(figsize=(10, 2 + 0.4 * w_2d.shape[0]))
    plt.imshow(w_2d, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(fraction=0.02)
    plt.xlabel("Time")
    plt.ylabel("Head")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def load_and_plot_file(npy_path: str):
    arr = np.load(npy_path, allow_pickle=False)
    base = os.path.basename(npy_path)
    stem, _ = os.path.splitext(base)
    folder = os.path.dirname(npy_path)

    if arr.ndim == 3 and arr.shape[1] == 1:
        # (B, 1, T) -> 배치별로 저장
        B, _, T = arr.shape
        for b in range(B):
            w = arr[b, 0]          # (T,)
            w2d = w[None, :]       # (1, T)
            out_name = f"{stem}.png" if B == 1 else f"{stem}_b{b}.png"
            out_path = os.path.join(folder, out_name)
            plot_cs_attn_heatmap_array(w2d, out_path, title=f"{stem} (b={b})" if B>1 else stem)
    elif arr.ndim == 2:
        # (1, T) 또는 (H, T) 케이스도 지원
        out_path = os.path.join(folder, f"{stem}.png")
        plot_cs_attn_heatmap_array(arr, out_path, title=stem)
    elif arr.ndim == 1:
        # (T,) -> (1, T)로 승격
        out_path = os.path.join(folder, f"{stem}.png")
        plot_cs_attn_heatmap_array(arr[None, :], out_path, title=stem)
    else:
        raise ValueError(f"Unsupported shape for {base}: {arr.shape}")

def main():
    folder = "/home/hice1/skim3513/scratch/causdiff/outputs/darai_l2/attention_map_20"
    pattern = "*.npy"
    paths = [p for p in glob.glob(os.path.join(folder, pattern)) if os.path.isfile(p)]
    paths = sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))

    if not paths:
        print(f"No .npy files found in {folder}")
        return

    print(f"Found {len(paths)} files.")
    for i, p in enumerate(paths):
        try:
            load_and_plot_file(p)
            print(f"[{i:03d}] OK  -> {os.path.basename(p)}")
        except Exception as e:
            print(f"[{i:03d}] ERR -> {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    main()
