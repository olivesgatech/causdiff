import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

L2_LABELS = [
    "Bake_pancake",
    "Cleaning_Countertops",
    "Cleaning_Floor",
    "Get_ingredients",
    "Having_a_meal",
    "Mix_ingredients",
    "Prep_ingredients",
    "Prepare_Kitchen_appliance",
    "Scroll_on_tablet",
    "Setting_a_table",
    "Take_out_Kitchen_and_cooking_tools",
    "Take_out_smartphone",
    "Throw_out_leftovers",
    "Using_Smartphone",
    "Using_Tablet",
    "Washing_and_Drying_dishes_with_hands",
    "UNDEFINED",
]

def normalize_rows(x, eps=1e-9):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def embed_labels_openclip(labels, device="cpu"):
    try:
        import torch, open_clip
        from torch.nn.functional import normalize as tnorm
    except Exception as e:
        raise RuntimeError("open_clip_torch is required for text embeddings. Install with `pip install open_clip_torch`.") from e

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    templates = [
        "A photo of someone {}.",
        "A person {} in a kitchen.",
        "A video frame of someone {}.",
        "Someone is {}.",
        "A cooking-related scene: {}.",
    ]
    all_embs = []
    with torch.no_grad():
        for lb in labels:
            texts = [t.format(lb.replace("_"," ").lower()) for t in templates]
            toks = tokenizer(texts).to(device)
            feats = model.encode_text(toks)
            feats = tnorm(feats.float(), dim=-1)
            e = tnorm(feats.mean(0, keepdim=True), dim=-1)  # [1,D]
            all_embs.append(e.cpu().numpy())
    return np.concatenate(all_embs, axis=0)  # [C,D]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True, help=".npy file, shape [T,512] (CLIP image features)")
    ap.add_argument("--outdir", type=str, required=True, help="output directory")
    ap.add_argument("--label-emb", type=str, default=None, help="optional .npy file [C,512] for label embeddings; skips OpenCLIP")
    ap.add_argument("--no-legend", action="store_true", help="do not create the legend image")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    X = np.load(args.features)  # [T,512]
    assert X.ndim == 2 and X.shape[1] == 512, f"Expected [T,512], got {X.shape}"
    T = X.shape[0]
    Xn = normalize_rows(X)

    if args.label_emb is not None:
        label_embs = np.load(args.label_emb)
        assert label_embs.shape[1] == 512, "Label embedding must have dim 512"
        C = label_embs.shape[0]
        labels = [f"class_{i}" for i in range(C)] if C != len(L2_LABELS) else L2_LABELS
    else:
        labels = L2_LABELS
        # Use OpenCLIP to embed label texts
        device = "cuda:1" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
        label_embs = embed_labels_openclip(labels, device=device)  # [C,512]

    # cosine similarity
    print(Xn.shape, label_embs.shape)
    sims = Xn @ label_embs.T  # [T,C]
    top_idx = sims.argmax(axis=1)  # [T]
    top_scores = sims[np.arange(T), top_idx]

    # save csv
    csv_path = os.path.join(args.outdir, "time_to_label.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time_index,label_index,score\n")
        for t, (idx, sc) in enumerate(zip(top_idx, top_scores)):
            f.write(f"{t},{idx},{float(sc):.6f}\n")

    # color visualization (no text on the main figure)
    C = label_embs.shape[0]
    cmap = plt.get_cmap("tab20")
    palette = np.array([cmap(i % 20)[:3] for i in range(C)])
    color_strip = palette[top_idx]  # [T,3]
    H = 40
    img = np.tile(color_strip[None, :, :], (H, 1, 1))

    plt.figure(figsize=(12, 1.2))
    plt.imshow(img, aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    fig_path = os.path.join(args.outdir, "l2_color_strip.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # optional legend (separate image so the main visualization has no text)
    if not args.no_legend:
        fig = plt.figure(figsize=(4, 0.3 * C))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        for i in range(C):
            ax.barh(i, 1.0, left=0, height=0.8, color=palette[i])
        ax.set_yticks(range(C))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks([])
        ax.set_xlim(0, 1.0)
        legend_path = os.path.join(args.outdir, "l2_color_legend.png")
        fig.savefig(legend_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Done.")
    print("Saved:", fig_path)
    if not args.no_legend:
        print("Saved legend:", legend_path)
    print("Saved CSV:", csv_path)

if __name__ == "__main__":
    main()
