import os
import numpy as np
import matplotlib.pyplot as plt

L2_LABELS = [
    "Bake_pancake","Cleaning_Countertops","Cleaning_Floor","Get_ingredients",
    "Having_a_meal","Mix_ingredients","Prep_ingredients","Prepare_Kitchen_appliance",
    "Scroll_on_tablet","Setting_a_table","Take_out_Kitchen_and_cooking_tools",
    "Take_out_smartphone","Throw_out_leftovers","Using_Smartphone","Using_Tablet",
    "Washing_and_Drying_dishes_with_hands","UNDEFINED",
]

def render_l2_from_subgoal_embeddings(
    subgoal_embeddings,
    outfile,
    outdir='./src/visualize-causal-attention/',
    label_embs=None,          # Optional np.ndarray [C, D]; if None, uses OpenCLIP text encodings
    labels=None,              # Optional list of label strings; default L2_LABELS (len must match C)
    embed_dim=512,            # D; also used to reshape if 1-D input provided
    make_legend=True,         # Whether to save the legend image
    openclip_arch="ViT-B-32", # OpenCLIP model for text embeddings (if label_embs is None)
    openclip_ckpt="openai",
    device="cuda"               # "cuda" or "cpu"; auto if None
):
    """
    Generate:
      - time_to_label.csv : top-1 L2 label over time with cosine scores
      - l2_color_strip.png: color-coded strip (time x color)
      - l2_color_legend.png: mapping color -> label (optional)

    Args:
      subgoal_embeddings: np.ndarray of shape [T, D] or flat [T*D]
      outdir: output directory path (created if missing)
      label_embs: optional np.ndarray [C, D] of label embeddings; if None we embed L2 labels with OpenCLIP
      labels: list of label strings (length C). Defaults to L2_LABELS if C matches.
      embed_dim: embedding dimensionality D (default 512)
      make_legend: save legend image (True/False)
      openclip_arch/openclip_ckpt/device: settings for OpenCLIP when label_embs is None
    Returns:
      dict with paths: {"csv", "strip", "legend"(optional)}
    """
    os.makedirs(outdir, exist_ok=True)

    # --- normalize helper ---
    def normalize_rows(x, eps=1e-9):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + eps)

    # --- load/reshape embeddings X: [T, D] ---
    X = np.asarray(subgoal_embeddings)
    if X.ndim == 1:
        assert X.size % embed_dim == 0, f"Flat input length {X.size} not divisible by embed_dim={embed_dim}"
        X = X.reshape(-1, embed_dim)
    assert X.ndim == 2 and X.shape[1] == embed_dim, f"Expected [T,{embed_dim}], got {X.shape}"
    T, D = X.shape
    Xn = normalize_rows(X)

    # --- obtain label embeddings [C, D] ---
    if label_embs is None:
        # fall back to OpenCLIP text embeddings
        try:
            import torch, open_clip
            from torch.nn.functional import normalize as tnorm
        except Exception as e:
            raise RuntimeError(
                "label_embs=None requires open_clip_torch. Install with `pip install open_clip_torch`."
            ) from e

        if device is None:
            device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or
                                (torch.cuda.is_available())) else "cpu"

        model, _, _ = open_clip.create_model_and_transforms(openclip_arch, pretrained=openclip_ckpt)
        tokenizer = open_clip.get_tokenizer(openclip_arch)
        model = model.to(device).eval()

        if labels is None:
            labels = L2_LABELS
        else:
            assert isinstance(labels, (list, tuple)) and len(labels) > 0

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
                e = tnorm(feats.mean(0, keepdim=True), dim=-1)  # [1,D']
                all_embs.append(e.cpu().numpy())
        label_embs = np.concatenate(all_embs, axis=0)  # [C, D']
        if label_embs.shape[1] != D:
            # If OpenCLIP dimension != subgoal D, project to common dim by PCA on label_embs then apply to X
            # Simpler: raise with a clear message.
            raise ValueError(f"Label embedding dim {label_embs.shape[1]} != subgoal dim {D}. "
                             "Use matching encoders or provide `label_embs` in the same dimension.")
    else:
        label_embs = np.asarray(label_embs)
        assert label_embs.ndim == 2 and label_embs.shape[1] == D, \
            f"label_embs must be [C,{D}], got {label_embs.shape}"
        if labels is None:
            labels = L2_LABELS if label_embs.shape[0] == len(L2_LABELS) else [f"class_{i}" for i in range(label_embs.shape[0])]
        else:
            assert len(labels) == label_embs.shape[0], "labels length must match C of label_embs"

    # --- cosine similarity & top-1 over time ---
    Ln = normalize_rows(label_embs)
    sims = Xn @ Ln.T                    # [T, C]
    top_idx = sims.argmax(axis=1)       # [T]
    top_scores = sims[np.arange(T), top_idx]

    # --- save CSV ---
    csv_path = os.path.join(outdir, f"{outfile}_time_to_label.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time_index,label_index,score\n")
        for t, (idx, sc) in enumerate(zip(top_idx, top_scores)):
            f.write(f"{t},{int(idx)},{float(sc):.6f}\n")

    # --- color strip (no text) ---
    C = Ln.shape[0]
    cmap = plt.get_cmap("tab20")
    palette = np.array([cmap(i % 20)[:3] for i in range(C)])
    color_strip = palette[top_idx]  # [T,3]
    H = 40
    img = np.tile(color_strip[None, :, :], (H, 1, 1))

    plt.figure(figsize=(12, 1.2))
    plt.imshow(img, aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    strip_path = os.path.join(outdir, f"{outfile}_l2_color_strip.png")
    plt.savefig(strip_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # --- legend (optional) ---
    legend_path = None
    if make_legend:
        fig = plt.figure(figsize=(4, 0.3 * C))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        for i in range(C):
            ax.barh(i, 1.0, left=0, height=0.8, color=palette[i])
        ax.set_yticks(range(C))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xticks([])
        ax.set_xlim(0, 1.0)
        legend_path = os.path.join(outdir, f"{outfile}_l2_color_legend.png")
        fig.savefig(legend_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    #return {"csv": csv_path, "strip": strip_path, **({"legend": legend_path} if legend_path else {})}
