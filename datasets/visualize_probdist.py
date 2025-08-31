import torch, open_clip
from torch.nn.functional import normalize
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt

# labels (L2)
labels = [
"Bake_pancake","Cleaning_Countertops","Cleaning_Floor","Get_ingredients",
"Having_a_meal","Mix_ingredients","Prep_ingredients","Prepare_Kitchen_appliance",
"Scroll_on_tablet","Setting_a_table","Take_out_Kitchen_and_cooking_tools",
"Take_out_smartphone","Throw_out_leftovers","Using_Smartphone","Using_Tablet",
"Washing_and_Drying_dishes_with_hands","UNDEFINED"
]

prompt_templates = [
"A photo of someone {}.",
"A person {} in a kitchen.",
"A video frame of {}.",
"Someone is {}.",
"A cooking-related scene: {}."
]

device = "cuda:1" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')
model = model.to(device).eval()

def clip_text_embed(texts):
    with torch.no_grad():
        toks = tokenizer(texts).to(device)
        feats = model.encode_text(toks)
        feats = normalize(feats.float(), dim=-1)
    return feats

# prompt-ensemble: average embeddings across templates
def embed_label_with_templates(label):
    texts = [t.format(label.replace('_', ' ').lower()) for t in prompt_templates]
    embs = clip_text_embed(texts)              # [T, D]
    return normalize(embs.mean(0, keepdim=True), dim=-1)  # [1, D]

# label matrix [C, D]
label_embs = torch.cat([embed_label_with_templates(c) for c in labels], dim=0)
label_embs.shape  # -> [17, D]

S = (label_embs @ label_embs.T).cpu()  # cosine similarity [C,C]
# For each label, you can build a neighbor distribution:
tau_neighbors = 0.05
neighbor_probs = torch.softmax(S / tau_neighbors, dim=-1)  # each row sums to 1


X = label_embs.cpu().numpy()

# PCA
pca = PCA(n_components=2, random_state=0).fit_transform(X)

# UMAP (often separates semantics better)
umap2d = UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=0).fit_transform(X)

def scatter_2d(Z, title, png_name):
    plt.figure()
    plt.scatter(Z[:,0], Z[:,1])
    for i, name in enumerate(labels):
        plt.text(Z[i,0], Z[i,1], name, fontsize=9)
    plt.title(title); plt.axis('off'); plt.tight_layout(); plt.savefig(f'/home/seulgi/work/causdiff/datasets/darai/{png_name}')

scatter_2d(pca, "L2 labels in CLIP space (PCA-2D)", 'pca')
scatter_2d(umap2d, "L2 labels in CLIP space (UMAP-2D)", 'umap2d')