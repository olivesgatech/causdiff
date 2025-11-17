# save as: plot_attention_folder.py
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def plot_cs_attn_heatmap_array(w_2d: np.ndarray, save_path: str, title="CS attention"):
    """
    w_2d: (H, T) 형태의 ndarray. (여기서는 H=헤드 수)
    """
    if w_2d.ndim != 2:
        raise ValueError(f"Expected 2D array (H,T), got shape {w_2d.shape}")
    plt.figure(figsize=(5, 1.5 + 0.01 * w_2d.shape[0]))
    plt.imshow(w_2d, aspect="auto", interpolation="nearest", cmap="afmhot")
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

    # (B, 1, T)인 경우: B를 헤드(H)로 간주하여 (H, T) = (B, T)로 합쳐 1장 그림
    if arr.ndim == 3 and arr.shape[1] == 1:
        # e.g., (5, 1, T) -> (5, T)
        w2d = arr[:, 0, :]
        out_path = os.path.join(folder, f"{stem}.png")
        plot_cs_attn_heatmap_array(w2d, out_path, title=f"{stem} (H={w2d.shape[0]})")

    # (B, T)로 이미 저장된 경우도 하나의 히트맵으로 처리
    elif arr.ndim == 2:
        # 두 가지 케이스 모두 지원: (H, T) 또는 (1, T)
        out_path = os.path.join(folder, f"{stem}.png")
        plot_cs_attn_heatmap_array(arr, out_path, title=f"{stem} (H={arr.shape[0]})")

    # (T,) -> (1, T)로 승격
    elif arr.ndim == 1:
        out_path = os.path.join(folder, f"{stem}.png")
        plot_cs_attn_heatmap_array(arr[None, :], out_path, title=stem)

    else:
        raise ValueError(f"Unsupported shape for {base}: {arr.shape}")

def main():
    folder = "/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/attention_map_30"
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
