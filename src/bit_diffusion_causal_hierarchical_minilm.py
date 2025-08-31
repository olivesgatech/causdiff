"""
https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import clip
from transformers import AutoTokenizer, AutoModel
import os, re, glob
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])#, "pred_goal_noise", "pred_goal_start"])

# BREAKFAST_GOAL = {
#     0: "a person is making scrambled egg",
#     1: "a person is making tea",
#     2: "a person is making coffee",
#     3: "a person is making milk",
#     4: "a person is making fried egg",
#     5: "a person is making cereals",
#     6: "a person is making pancake",
#     7: "a person is making sandwich",
#     8: "a person is making salat",
#     9: "a person is making juice"
# }

BREAKFAST_GOAL = {
    0: "the goal is to prepare scrambled eggs for breakfast",
    1: "the goal is to brew tea for breakfast",
    2: "the goal is to brew coffee for breakfast",
    3: "the goal is to pour milk for breakfast",
    4: "the goal is to prepare fried eggs for breakfast",
    5: "the goal is to prepare a bowl of cereal for breakfast",
    6: "the goal is to cook pancakes for breakfast",
    7: "the goal is to make a sandwich for breakfast",
    8: "the goal is to prepare a salad for breakfast",
    9: "the goal is to make juice for breakfast"
}


BREAKFAST_ACTION = {
    0:  "a person is silent",
    1:  "a person is pouring cereals",
    2:  "a person is pouring milk",
    3:  "a person is stirring cereals",
    4:  "a person is taking a bowl",
    5:  "a person is pouring coffee",
    6:  "a person is taking a cup",
    7:  "a person is spooning sugar",
    8:  "a person is stirring coffee",
    9:  "a person is pouring sugar",
    10: "a person is pouring oil",
    11: "a person is cracking an egg",
    12: "a person is adding salt and pepper",
    13: "a person is frying an egg",
    14: "a person is taking a plate",
    15: "a person is putting an egg on a plate",
    16: "a person is taking eggs",
    17: "a person is buttering a pan",
    18: "a person is taking a knife",
    19: "a person is cutting an orange",
    20: "a person is squeezing an orange",
    21: "a person is pouring juice",
    22: "a person is taking a glass",
    23: "a person is taking a squeezer",
    24: "a person is spooning powder",
    25: "a person is stirring milk",
    26: "a person is spooning flour",
    27: "a person is stirring dough",
    28: "a person is pouring dough into a pan",
    29: "a person is frying a pancake",
    30: "a person is putting a pancake on a plate",
    31: "a person is pouring flour",
    32: "a person is cutting fruit",
    33: "a person is putting fruit into a bowl",
    34: "a person is peeling fruit",
    35: "a person is stirring fruit",
    36: "a person is cutting a bun",
    37: "a person is smearing butter",
    38: "a person is taking a topping",
    39: "a person is putting the topping on top",
    40: "a person is putting the bun together",
    41: "a person is taking butter",
    42: "a person is stirring eggs",
    43: "a person is pouring eggs into a pan",
    44: "a person is stir-frying eggs",
    45: "a person is adding a teabag",
    46: "a person is pouring water",
    47: "a person is stirring tea"
}

DARAI_GOAL = {
    0: "Using handheld smart devices",
    1: "Making a cup of instant coffee",
    2: "Making pancake",
    3: "Dining",
    4: "Cleaning dishes",
    5: "Making a cup of coffee in coffee maker",
    6: "Cleaning the kitchen",
}
DARAI_ACTION = {
    0: "Add batter",
    1: "Add coffee",
    2: "Add flour",
    3: "Add milk",
    4: "Add sugar",
    5: "Add water",
    6: "Check cabinet",
    7: "Check pancake",
    8: "Check refrigerator",
    9: "Clean with broom",
    10: "Clean with mop",
    11: "Clean with paper towel",
    12: "Clean with towel",
    13: "Conversation on the phone",
    14: "Crack egg",
    15: "Drink",
    16: "Dry dishes",
    17: "Eat",
    18: "Fill coffee machine with water",
    19: "Fill kettle with water",
    20: "Get coffee",
    21: "Get cup",
    22: "Get filter",
    23: "Get instant coffee",
    24: "Get pan",
    25: "Get spoon",
    26: "Load dishwasher",
    27: "Place cup",
    28: "Place dishes",
    29: "Place drink",
    30: "Place filter",
    31: "Place food",
    32: "Place pan",
    33: "Place silverware",
    34: "Prepare for activity",
    35: "Rinse dishes",
    36: "Scroll on the phone",
    37: "Scroll on the tablet",
    38: "Stir",
    39: "Stir pancake ingredients",
    40: "Take out Kitchen and cooking tools",
    41: "Take out pancake ingredients",
    42: "Turn on coffee machine",
    43: "Turn on dishwasher",
    44: "Turn on kettle",
    45: "Turn on stove",
    46: "Unloading dishwasher",
    47: "UNDEFINED",
}

SALADS_GOAL = {
    0: "Cut and mix ingredients",
    1: "Prepare dressing",
    2: "Serve salad",
    3: "Action end",
    4: "Action start",
}

def causal_attention_summary(subgoal_seq: torch.Tensor) -> torch.Tensor:
    """
    Efficient time-causal self-attention summarization over (B, T, D).
    Each timestep attends only to previous and current subgoals.
    
    Args:
        subgoal_seq: (B, T, D) tensor of subgoal sequence
    
    Returns:
        context: (B, T, D) causal attention summary
    """
    B, T, D = subgoal_seq.shape
    q = subgoal_seq  # (B, T, D)
    k = subgoal_seq
    v = subgoal_seq

    # Compute attention scores: (B, T, T)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, T, T)

    # Apply causal mask (upper triangle = -inf)
    causal_mask = torch.triu(torch.ones(T, T, device=subgoal_seq.device), diagonal=1).bool()
    attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)

    # Weighted sum over values: (B, T, D)
    context = torch.matmul(attn_weights, v)

    return context

# def causal_attention_summary(subgoal_seq: torch.Tensor, h: int = 4) -> torch.Tensor:
#     B, T, D = subgoal_seq.shape
#     device = subgoal_seq.device
#     context = []

#     for t in range(T):
#         t_start = max(0, t - h)
#         q = subgoal_seq[:, t:t+1, :]               # (B, 1, D)
#         k = subgoal_seq[:, t_start:t+1, :]         # (B, h', D)
#         v = subgoal_seq[:, t_start:t+1, :]         # (B, h', D)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, 1, h')
#         attn_weights = F.softmax(attn_scores, dim=-1)                    # (B, 1, h')
#         attended = torch.matmul(attn_weights, v)                         # (B, 1, D)

#         context.append(attended.squeeze(1))  # → (B, D)

#     context = torch.stack(context, dim=1)  # (B, T, D)
#     return context

def encode_texts(texts, clip_model, device='cuda:1'):
    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)  # (N, 512)
        text_features = F.normalize(text_features.float(), dim=-1)
    return text_features

def decode_clip_embedding_to_text(
    clip_embedding,
    device='cuda:1',
    features_dir="/home/hice1/skim3513/scratch/causdiff/datasets/darai/features_description_text",
    out_txt_path="/home/hice1/skim3513/scratch/causdiff/datasets/darai/out.txt", topk=5
):
    # ---- 1) clip_embedding → (1, D) float tensor on device ----
    if not isinstance(clip_embedding, torch.Tensor):
        clip_embedding = torch.as_tensor(clip_embedding)
    clip_embedding = clip_embedding.float().to(device)

    if clip_embedding.ndim == 1:
        clip_embedding = clip_embedding.unsqueeze(0)  # (1, D)
    else:
        clip_embedding = clip_embedding.reshape(clip_embedding.shape[0], -1)
        if clip_embedding.shape[0] > 1:
            clip_embedding = clip_embedding[:1, :]
    clip_embedding = F.normalize(clip_embedding, dim=-1)  # (1, D)

    # ---- 2) feature 로드 → (N, D) ----
    npy_paths = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
    if not npy_paths:
        raise FileNotFoundError(f"No .npy features found in {features_dir}")

    feats_list, stems = [], []
    for p in npy_paths:
        arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim >= 2:
            arr = arr.reshape(-1)
        if arr.size == 0:
            continue
        feats_list.append(arr)
        stems.append(os.path.splitext(os.path.basename(p))[0])  # "00017"

    if not feats_list:
        raise RuntimeError(f"No valid feature arrays in {features_dir}")

    feats = torch.from_numpy(np.stack(feats_list, axis=0)).to(device)  # (N, D)
    feats = F.normalize(feats, dim=-1)

    # ---- 3) 코사인 유사도 & Top-K ----
    sims = torch.matmul(clip_embedding, feats.t()).flatten()  # (N,)
    k = int(min(topk, sims.numel()))
    top_vals, top_idxs = torch.topk(sims, k=k, largest=True, sorted=True)  # (k,)

    # ---- 4) out.txt 로부터 텍스트 매핑 ----
    with open(out_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.rstrip('\n') for ln in f]

    results = []
    for rank in range(k):
        idx = int(top_idxs[rank].item())
        stem = stems[idx]                  # "00017"
        feat_path = npy_paths[idx]
        # stem에서 숫자 추출 후 +1 → 1-indexed 라인
        m = re.search(r'(\d+)$', stem)
        file_index = int(m.group(1)) if m else idx
        line_num = file_index + 1
        if line_num < 1: line_num = 1
        if line_num > len(lines): line_num = len(lines)
        text = lines[line_num - 1].strip()

        results.append({
            "text": text,
            "score": float(top_vals[rank].item()),
        })

    return results

def pick_darai_action_by_similarity(model_out, clip_model, device="cuda"):
    """
    model_out: torch.Tensor or np.ndarray, shape (512,) or (1,512) or (N,512)
    return: dict (top-1) 또는 list of dicts (top-k)
            {"index": int, "label": str, "score": float}
    """
    # 1) 텍스트 임베딩
    action_ids = sorted(DARAI_ACTION.keys())
    action_texts = [DARAI_ACTION[i] for i in action_ids]
    action_feats = encode_texts_clip(action_texts, clip_model, device)  # (A,512)

    # 2) model_out 정규화 & 2D화
    if not isinstance(model_out, torch.Tensor):
        model_out = torch.as_tensor(model_out)
    model_out = model_out.float().to(device)
    if model_out.ndim == 1:
        model_out = model_out.unsqueeze(0)            # (1,512)
    else:
        model_out = model_out.reshape(model_out.shape[0], -1)
        if model_out.shape[0] > 1:
            model_out = model_out[:1, :]              # 첫 샘플만 사용
    model_out = F.normalize(model_out, dim=-1)        # (1,512)

    # 3) 코사인 유사도 (정규화 → 내적 = cosine)
    sims = torch.matmul(model_out, action_feats.t()).squeeze(0)  # (A,)
    top_vals, top_idxs = torch.topk(sims, k=1, largest=True, sorted=True)
    idx = int(top_idxs[r].item())
    return action_ids[idx]

def build_action_feats(DARAI_ACTION, clip_model, device="cuda"):
    import clip
    texts = [DARAI_ACTION[i] for i in sorted(DARAI_ACTION.keys())]
    with torch.no_grad():
        toks = clip.tokenize(texts).to(device)
        feats = clip_model.encode_text(toks).float()
    return F.normalize(feats, dim=-1)  # (A,512)

# def ce_loss_with_clip_head(
#     model_out,
#     action_feats,
#     targets,
#     temperature: float = 0.07,
#     mask_all=None,                   # ⬅️ (옵션) 마스크. shape은 targets와 동일한 선행축 (B) 또는 (S,B,T) 등
#     label_smoothing: float = 0.0,
#     micro_average: bool = True,      # True면 전체 마스크 합으로 나눠 마이크로 평균
# ):
#     """
#     model_out: (..., 512)  - 예: (B,512) 또는 (B,T,512) 또는 (S,B,T,512)
#     action_feats: (A,512)  - build_action_feats()로 준비
#     targets:     (...)     - model_out의 선행축과 동일. 예: (B,) 또는 (B,T) 또는 (S,B,T) [LongTensor]
#     mask_all:    (...)     - targets와 브로드캐스트 호환되는 마스크 (0/1). 예: (B,) / (B,T) / (S,B,T)
#     반환: (loss: scalar, pred: targets.shape의 예측 인덱스)
#     """
#     D = action_feats.size(-1)
#     dev = action_feats.device
#     x = model_out
#     if not isinstance(x, torch.Tensor):
#         x = torch.as_tensor(x)
#     x = x.to(dev).float()
#     x = x.view(-1, D)
#     x = F.normalize(x, dim=-1)                        # (N,512)

#     logits = x @ action_feats.to(dev).t() / float(temperature)   # (N,A)

#     tgt = torch.as_tensor(targets, device=dev)
#     tgt = tgt.argmax(dim=-1)
#     tgt = tgt.reshape(-1).long()             # (N,)
    
#     loss_vec = F.cross_entropy(
#         logits, tgt, reduction='none', label_smoothing=label_smoothing
#     )                                                 # (N,)

#     if mask_all is not None:
#         m = torch.as_tensor(mask_all, device=dev, dtype=loss_vec.dtype).reshape(-1)  # (N,)
#         masked = loss_vec * m
#         if micro_average:
#             denom = m.sum().clamp_min(1e-6)
#             loss = masked.sum() / denom
#         else:
#             # 필요 시 per-sample 평균으로 바꾸고 싶으면 여기서 그룹 단위로 나눠도 됨
#             denom = (m > 0).float().sum().clamp_min(1e-6)
#             loss = masked.sum() / denom
#     else:
#         loss = loss_vec.mean()

#     pred_flat = logits.argmax(dim=-1)            # (N,)
#     pred = pred_flat.reshape(*tgt.shape) 

#     return loss#, pred

def ce_loss_with_clip_head(
    model_out,
    action_feats,                  # (A,512)  = DARAI_ACTION 텍스트를 CLIP으로 임베딩한 프로토타입
    targets,                       # (...,512) 임베딩  또는 (...,A) 원-핫  또는 (...,) 인덱스
    temperature: float = 0.07,
    mask_all=None,                 # (같은 leading shape) 마스크 0/1
    label_smoothing: float = 0.0,
    micro_average: bool = True,
    class_mask=None,               # (K,) 허용 클래스 인덱스. None이면 0..A-1 모두 허용
    debug: bool = False,
):
    """
    model_out: (..., 512)           e.g. (S,B,T,512)
    action_feats: (A,512)           A=48
    targets:   (...,512) 임베딩 or (...,A) 원-핫 or (...,) 인덱스
    class_mask: 허용할 클래스의 인덱스 리스트/LongTensor. 제공 시 해당 subset에 대해서만 CE 계산.
    반환: (loss, pred)  pred의 shape = targets의 leading shape (클래스 축 제외)
    """
    dev = action_feats.device
    A, D = action_feats.shape
    leading_shape = tuple(model_out.shape[:-1])        # ex) (S,B,T)
    N = int(torch.tensor(leading_shape).clamp_min(1).prod().item()) if len(leading_shape) else model_out.shape[0]

    # --- 1) 로짓: (N,A) ---
    x = torch.as_tensor(model_out, device=dev, dtype=torch.float32).reshape(-1, D)
    x = F.normalize(x, dim=-1)
    proto = F.normalize(action_feats, dim=-1)          # (A,512)
    logits_full = x @ proto.to(dev).t() / float(temperature)  # (N,A)

    # --- 2) 타깃을 "정수 인덱스"로 통일 ---
    tgt = torch.as_tensor(targets, device=dev)
    if tgt.shape == leading_shape + (D,):              # (...,512) 임베딩 → 최근접 클래스
        t = F.normalize(tgt.reshape(-1, D).float(), dim=-1)       # (N,512)
        tgt_logits = t @ proto.to(dev).t() / float(temperature)   # (N,A)
        tgt_idx = tgt_logits.argmax(dim=-1)                       # (N,)
    elif tgt.shape == leading_shape + (A,):            # (...,A) 원-핫/소프트 → argmax
        tgt_idx = tgt.argmax(dim=-1).reshape(-1).long()           # (N,)
    else:                                              # (...,) 인덱스
        tgt_idx = tgt.reshape(-1).long()                            

    # 길이 맞추기(안전 가드)
    if tgt_idx.numel() != logits_full.size(0):
        minN = min(tgt_idx.numel(), logits_full.size(0))
        tgt_idx    = tgt_idx[:minN]
        logits_full = logits_full[:minN]
        if mask_all is not None:
            mask_all = torch.as_tensor(mask_all, device=dev)
            mask_all = mask_all.reshape(-1)[:minN]

    # --- 3) 클래스 마스크(선택): 특정 클래스만 허용하고 싶을 때 ---
    if class_mask is not None:
        allow = torch.as_tensor(class_mask, device=dev, dtype=torch.long)  # (K,)
        # 로짓을 허용된 클래스 열만 고름 → (N,K)
        logits = logits_full.index_select(dim=1, index=allow)

        # 원래 인덱스를 subset 인덱스로 매핑, 허용 외는 ignore_index로 마스킹
        ignore_index = -100
        map_table = -torch.ones(A, device=dev, dtype=torch.long)  # default -1
        map_table[allow] = torch.arange(allow.numel(), device=dev)
        tgt_mapped = map_table[tgt_idx]                           # (N,)
        # 허용되지 않은 타깃은 무시되도록 CE(ignore_index) 사용
        loss_vec = F.cross_entropy(
            logits, tgt_mapped, reduction='none',
            label_smoothing=label_smoothing, ignore_index=ignore_index
        )
        # 예측 (subset 내 argmax → 원래 인덱스로 복원)
        pred_idx = allow[logits.argmax(dim=-1)]
    else:
        logits = logits_full                                     # (N,A)
        loss_vec = F.cross_entropy(
            logits, tgt_idx, reduction='none', label_smoothing=label_smoothing
        )
        pred_idx = logits.argmax(dim=-1)                         # (N,)

    # --- 4) 마스크 적용 & 리덕션 ---
    if mask_all is not None:
        m = torch.as_tensor(mask_all, device=dev, dtype=loss_vec.dtype).reshape(-1)
        minN = min(m.numel(), loss_vec.numel())
        m, loss_vec, pred_idx = m[:minN], loss_vec[:minN], pred_idx[:minN]
        if micro_average:
            loss = (loss_vec * m).sum() / m.sum().clamp_min(1e-6)
        else:
            loss = (loss_vec * m).sum() / (m > 0).float().sum().clamp_min(1e-6)
    else:
        loss = loss_vec.mean()

    # --- 5) 예측을 원래 leading shape로 복원 ---
    pred = pred_idx.reshape(*leading_shape) if len(leading_shape) else pred_idx
    return loss


# # Main function: input = CLIP vector → output = "goal_text, action_text"
# def decode_clip_embedding_to_text(clip_embedding, clip_model, device='cuda:1'):
#     if clip_embedding.dim() == 1:
#         clip_embedding = clip_embedding.unsqueeze(0)  # (1, 512)
#     clip_embedding = F.normalize(clip_embedding.float(), dim=-1)  # normalize

#     # Encode candidates
#     goal_texts = list(BREAKFAST_GOAL.values())
#     action_texts = list(BREAKFAST_ACTION.values())
#     goal_feats = encode_texts(goal_texts, clip_model, device)
#     action_feats = encode_texts(action_texts, clip_model, device)

#     # Compute cosine similarity
#     goal_sim = clip_embedding @ goal_feats.T  # (1, 10)
#     action_sim = clip_embedding @ action_feats.T  # (1, 48)

#     # Pick top-1 from each
#     # goal_idx = goal_sim.argmax(dim=-1).item()
#     # action_idx = action_sim.argmax(dim=-1).item()
#     # goal_text = goal_texts[goal_idx]
#     # action_text = action_texts[action_idx]
#     topk=5
#     goal_vals, goal_idxs = goal_sim.topk(topk, dim=-1)
#     action_vals, action_idxs = action_sim.topk(topk, dim=-1)

#     topk_goals = [(goal_texts[goal_idxs[0][i]], goal_vals[0][i].item()) for i in range(topk)]
#     topk_actions = [(action_texts[action_idxs[0][i]], action_vals[0][i].item()) for i in range(topk)]

#     return topk_goals, topk_actions


def cosine_loss(x, y):
    cos_sim = F.cosine_similarity(x, y, dim=-1)
    return 1 - cos_sim

class DiffusionModel(nn.Module):
    """
    Template model for the diffusion process
    """

    def __init__(
        self,
    ):
        super().__init__()


    def forward(self, X_0, X_t, batch):
        raise NotImplementedError("Nope")


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # B x 1 x 1 


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def l2norm(t):
    return F.normalize(t, dim=-1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianBitDiffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        goalmodel:nn.Module,
        condition_x0:False,
        clip_model,
        *,
        num_classes=48,
        timesteps=1000,  
        ddim_timesteps=100,
        betas=None,
        loss_type="l2",
        objective="pred_x0",
        beta_schedule="cosine",
    ):
        
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to('cuda:1')
        self.text_encoder.eval()  # inference only
        
        print(f'Num classes : {num_classes}')
        print(f'Loss type : {loss_type}')
        print(f'Objective: {objective}')
        print(f'Beta schedule : {beta_schedule}')

        self.model = model
        self.num_classes = num_classes
        self.condition_x0 = condition_x0
        self.goalmodel = goalmodel

        # RECONSTRUCTION OBJ
        self.objective = objective
        self.loss_type = loss_type

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0(predict image start)"  # noqa E501


        # VARIANCE
        if betas is None:
            if beta_schedule == "linear":
                betas = linear_beta_schedule(timesteps)
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(timesteps)
            else:
                raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


        # SAMPLING
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # PARAMS
        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # FORWARD DIFFUSION q(x_t | x_{t-1}) 
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))


        # POSTERIOR q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)),)
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),)
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),)

        # DDIM
        self.eta = 0.
        c = timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, timesteps, c)))
        
        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq

        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False

        self.alpha = nn.Parameter(torch.full((1,1,512), 0.0))
        self.action_feats = build_action_feats(DARAI_ACTION, self.clip)
        self.id2text = [DARAI_ACTION[i] for i in range(len(DARAI_ACTION))]
        self.emb_norm = nn.LayerNorm(512, elementwise_affine=True)
    
    def semantic_consistency_loss(self, subgoal_features, global_goal):
        """
        subgoal_features: (S,B,T,D) - semantic features from DiffSingleStageModel
        global_goal: (B,C) - one-hot encoded global goal
        """
        S, B, T, D = subgoal_features.shape
        
        # Get global goal features (only once)
        #with torch.amp.autocast('cuda:1'):
            #global_goal_classes = global_goal.argmax(dim=-1)  # (B,)
            #global_goal_texts = [f"action {idx.item()}" for idx in global_goal_classes]
            #global_goal_texts = [BREAKFAST_GOAL[idx.item()] for idx in global_goal_classes]
            #global_goal_texts = [DARAI_GOAL[idx.item()] for idx in global_goal_classes]
            #global_goal_texts = [SALADS_GOAL[idx.item()] for idx in global_goal_classes]
            
            
            #global_goal_tokens = clip.tokenize(global_goal_texts).to(global_goal.device)
            #global_goal_features = self.clip.encode_text(global_goal_tokens)  # (B, 512)
            
            # Expand global goal features to match subgoal shape
            # (B, 512) -> (B, 1, 512) -> (B, T, 512)
            #global_goal_features = global_goal_features.unsqueeze(1).expand(-1, T, -1)
            # (B, T, 512) -> (1, B, T, 512) -> (S, B, T, 512)
            #global_goal_features = global_goal_features.unsqueeze(0).expand(S, -1, -1, -1)
        global_goal_features = global_goal.unsqueeze(0).expand(S,-1,-1,-1)
        
        # Now both tensors are (S, B, T, D)
        similarity = F.cosine_similarity(
            subgoal_features,
            global_goal_features,
            dim=-1  # compute similarity along feature dimension
        )  # (S, B, T)
        
        loss = 1 - similarity.mean()
            
        return loss
    
    @property
    def loss_fn(self):
        if self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
          - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
              extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )  



    # SAMPLE from q(x_t | x_o)
    def q_sample(self, x_start, t, noise=None):
        """
        :param x_start: {B x T x C}
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return ( extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

  
    # PREDICTION and LOSSES
    def p_losses(self,
                t,
                x_0,
                obs,
                mask_all,
                mask_past,gt_goal=None,gt_goal_one_hot=None,
                noise=None):
        condition = False
        # SAMPLE x_t from q(x_t | x_o)
        
        ############## GUESS LABEL ###################
        x_start = x_0
        ###############################################
        ############## GUESS SEMANTIC ###################
        # targets_idx_bt = x_0.argmax(dim=-1)
        # B, T = targets_idx_bt.shape
        # idx_flat = targets_idx_bt.reshape(-1).tolist()
        # texts_flat = [self.id2text[i] for i in idx_flat]
        # with torch.no_grad():
        #     toks = clip.tokenize(texts_flat).to(gt_goal.device)              # (B*T, ctx_len)
        #     feats = self.clip.encode_text(toks).float()              # (B*T, 512)
        #     feats = F.normalize(feats, dim=-1)

        # x_start = feats.view(B, T, -1)                               # (B,T,512)
        # x_start = self.emb_norm(x_start)
        ###############################################
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # sample: goal
        _, T, _ = gt_goal_one_hot.shape
        global_goal_classes = gt_goal_one_hot[:,-1].argmax(dim=-1)  # (B,)
        #global_goal_texts = [BREAKFAST_GOAL[idx.item()] for idx in global_goal_classes]
        global_goal_texts = [DARAI_GOAL[idx.item()] for idx in global_goal_classes]

        # ## minilm tokenization
        goal_inputs = self.tokenizer(global_goal_texts, return_tensors="pt", padding=True, truncation=True).to(gt_goal.device)
        with torch.no_grad():
            outputs = self.text_encoder(**goal_inputs)
            goal_features = outputs.last_hidden_state.mean(dim=1)  # (B, D), average pooling

        # CLIP Tokenization
        # goal_tokens = clip.tokenize(global_goal_texts).to(gt_goal.device)
        # goal_features = self.clip.encode_text(goal_tokens)  # (B, D_clip)

        # 2. 확장 (time axis 맞추기)
        goal_start = goal_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_clip)
        #goal_start = gt_goal_one_hot # (16, 1816, 10)
        goal_noise = None
        goal_noise = default(goal_noise, lambda: torch.randn_like(goal_start))
        
        goal_t = self.q_sample(x_start=goal_start, t=t, noise=goal_noise)

        # OBSERVATION CONDITIONING
        obs_cond = obs * mask_past

        # SELF-CONDITIONING
        #self_cond = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        self_cond = torch.zeros((gt_goal_one_hot.shape[0], gt_goal_one_hot.shape[1], 384), device=gt_goal_one_hot.device)#384
        
        if torch.rand((1)) < 0.5 and self.condition_x0:
            with torch.no_grad():
                _, infer_goal = self.goalmodel(
                    x=goal_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, # frames
                    self_cond=self_cond,
                )
                #self_cond = causal_attention_summary(infer_goal[-1].detach())
                
                self_cond = infer_goal[-1].detach()

        # REVERSE STEP
        _, model_out_goal = self.goalmodel(
            x=goal_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C

        subgoal_seq = model_out_goal # (S x B x T x C)
        #goal_logits = model_out_goal.mean(dim=2, keepdim=True) # (S, B, 1, C)
        gt_goal_one_hot = gt_goal_one_hot[:, :1, :]
        # gt_goal_one_hot = repeat(gt_goal_one_hot, 'b 1 c -> s b 1 c', s=goal_logits.shape[0])
        
        #loss_goal = self.loss_fn(goal_logits, gt_goal_one_hot, reduction="none")
        
        #self_cond = torch.zeros_like(x_0).to(x_0.device)
        #self_cond = torch.zeros_like(model_out_goal[-1]).to(model_out_goal.device)
        #self_cond = torch.zeros((model_out_goal.shape[1], model_out_goal.shape[2], model_out_goal.shape[3]), device=model_out_goal.device)
        # SELF-CONDITIONING
        if torch.rand((1)) < 0.5 and self.condition_x0:
            with torch.no_grad():
                self_cond = torch.zeros((model_out_goal.shape[1], model_out_goal.shape[2], x_0.shape[2]+model_out_goal.shape[3]), device=model_out_goal.device)
                self_cond, _ = self.model(
                    x=x_t, 
                    t=t, 
                    stage_masks=mask_all,
                    obs_cond=obs_cond, 
                    self_cond=self_cond
                )
                self_cond = self_cond[-1]
                self_cond = self_cond.detach()
        else:
            self_cond = torch.zeros_like(x_0).to(x_0.device)
            
        ## concat: 
        ## self_cond: (b,t,c)
        ## goal_logits[-1]: (b,1,c)
        #goal_repeat = goal_logits[-1].expand(-1, self_cond.shape[1], -1) # (b,1,c)->(b,t,c)
        goal_repeat = subgoal_seq[-1] # (B x T x C)
        #alpha = torch.sigmoid(self.alpha)
        #self_cond = alpha * self_cond + (1 - alpha) * goal_repeat
        self_cond = torch.cat([self_cond, goal_repeat], dim=2) # (b,t,2c)
        # REVERSE STEP
        model_out, _ = self.model(
            x=x_t,
            t=t,
            stage_masks=mask_all,
            obs_cond=obs_cond,
            self_cond=self_cond,
        )  # S x B x T x C

        # LOSS
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_0 
        else:
            raise ValueError(f"unknown objective {self.objective}")
        

        # KL (q(x_t-1 | x_o, x_t) || p(x_t-1 | x_t))
        if self.loss_type == 'l2':
            target = repeat(target, 'b t c -> s b t c', s=model_out.shape[0])
            
            mask_all = torch.stack(mask_all, dim=0)
            loss = self.loss_fn(model_out, target, reduction="none")  # S x B x T x C
            loss = torch.sum(torch.mean(loss * mask_all, dim=(2, 3)))
            #loss += ce_loss_with_clip_head(model_out, self.action_feats, target, mask_all=mask_all)
            if gt_goal is not None:
                ### goal diffusion: goal_logits (S x B x 1 x C)
                ### action diffusion: model_out (S x B x T x C)
                loss_goal = self.semantic_consistency_loss(
                    subgoal_features=subgoal_seq,
                    global_goal=goal_start
                )
                #loss_goal = torch.sum(torch.mean(loss_goal * mask_all, dim=(2, 3)))
                loss += loss_goal

        # OUT
        return loss, rearrange(model_out, 's b t c -> s b c t')



    def forward(self, batch, *args, **kwargs):
        gt_goal = batch['goal']
        gt_goal_one_hot = batch['goal_one_hot']
        x_0 = batch['x_0']  # class labels, (16, 2996, 48) # 48 is the class number
        obs = batch['obs']  # features, (16, 2996, 2048)
        
        masks_stages = batch['masks_stages'] # 5-length (16, 2996, 1) each
        masks_stages = [mask.to(torch.bool) for mask in masks_stages] # 5-length (16, 2996, 1) each
        
        mask_past = batch['mask_past'] # (16, 2996, 1)
        mask_past = mask_past.to(torch.bool)
        mask_past = repeat(mask_past, 'b t 1 -> b t c', c=obs.shape[-1]) # (16, 2996, 2048)

        # get random diff timestep
        t = torch.randint(0, self.num_timesteps, (obs.size(0),), device=obs.device).long() # (16)
        return self.p_losses(
            t = t,
            x_0 = x_0,
            obs = obs,
            mask_past = mask_past,
            mask_all = masks_stages,
            gt_goal=gt_goal, gt_goal_one_hot=gt_goal_one_hot,
            *args, **kwargs
        )

 

    # ---------------------------------- INFERENCE (DDIM) --------------------------------------

    def model_predictions(self, x, pred_x_start_prev, t, obs, stage_masks, gt_goal=None, gt_goal_one_hot=None):
        x_t = x
        B,T,C=gt_goal_one_hot.shape
        # Given x_t, reconsturct x_0
        #self_cond_goal = torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
        self_cond = torch.zeros_like(pred_x_start_prev).to(pred_x_start_prev.device)
        #self_cond = torch.zeros((B, T, 58), device=pred_x_start_prev.device)
        if self.condition_x0:
            self_cond = pred_x_start_prev
        
        ###################################################################################
        #goal_t = gt_goal_one_hot#torch.randn((B, T, C)).to(pred_x_start_prev.device)  # e.g., (16, 1, 48) shaped random noise
        _, T, _ = gt_goal_one_hot.shape
        global_goal_classes = gt_goal_one_hot[:,-1].argmax(dim=-1)  # (B,)
        #global_goal_texts = [BREAKFAST_GOAL[idx.item()] for idx in global_goal_classes]
        global_goal_texts = [DARAI_GOAL[idx.item()] for idx in global_goal_classes]
        # ## minilm tokenization
        goal_inputs = self.tokenizer(global_goal_texts, return_tensors="pt", padding=True, truncation=True).to(gt_goal.device)
        with torch.no_grad():
            outputs = self.text_encoder(**goal_inputs)
            goal_features = outputs.last_hidden_state.mean(dim=1)  # (B, D), average pooling
        goal_t = goal_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        
        ## CLIP tokenization
        # goal_tokens = clip.tokenize(global_goal_texts).to(gt_goal.device)
        # goal_features = self.clip.encode_text(goal_tokens)  # (B, D_clip)
        # goal_t = goal_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_clip)

        _, infer_goal = self.goalmodel(
            x=goal_t, 
            t=t, 
            stage_masks=stage_masks,
            obs_cond=obs,
            #self_cond=torch.zeros_like(gt_goal_one_hot).to(gt_goal_one_hot.device)
            self_cond = torch.zeros((gt_goal_one_hot.shape[0], gt_goal_one_hot.shape[1], 384), device=gt_goal_one_hot.device)#384
            #self_cond = self_cond
        )

        infer_goal = infer_goal[-1] # (25, 106, 512)
        # infer_goal: (T, 512)
        # for i in range(25):
        #     sim_matrix = cosine_similarity(infer_goal[i].cpu().numpy())
        #     plt.figure(figsize=(6, 5))
        #     img = plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
        #     plt.axis('off')            # ✅ 모든 축과 라벨 제거
        #     plt.title("")              # ✅ 제목 제거
        #     plt.xticks([])             # ✅ x축 눈금 제거
        #     plt.yticks([])             # ✅ y축 눈금 제거

        #     plt.tight_layout()
        #     plt.savefig(f'subgoal_{i}.png', dpi=150, bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # decoded_texts = []
        # # for i in range(25):
        # #     print(goal_features.shape, goal_features[0].shape)
        # text = decode_clip_embedding_to_text(goal_features[0])
        # print(f"///////////////// goal: {text} ///////////////////////")

        # # 텍스트 디코딩 + 저장
        # for i in range(infer_goal.shape[1]):
        #     clip_vec = infer_goal[0][i]
            
        #     text = decode_clip_embedding_to_text(clip_vec)
        #     decoded_texts.append(text)
        #     print(f"[{i}] {text}")

        # # 결과를 파일에 저장
        # output_path = "decoded_subgoals.txt"
        # with open(output_path, "w", encoding="utf-8") as f:
        #     for line in decoded_texts:
        #         f.write(str(line) + "\n")

        # print(f"\n✅ completed: {output_path}")

        
        self_cond = torch.cat([self_cond, infer_goal], dim=2)
        
        ##################################################################################
        
        # PRED
        model_output,_ = self.model(
            x=x_t,
            t=t,
            stage_masks=stage_masks,
            obs_cond=obs,
            self_cond=self_cond,
        )
        model_output = model_output[-1]
        
        if self.objective == "pred_noise":
            pred_noise = model_output 
            pred_x_start = self.predict_start_from_noise(x, t, pred_noise) * stage_masks[-1]
            
        elif self.objective == "pred_x0":
            pred_x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start) * stage_masks[-1]

        #pred_goal_start = infer_goal
        #pred_goal_noise = self.predict_noise_from_start(x, t, pred_goal_start) * stage_masks[-1]
            
        return ModelPrediction(pred_noise, pred_x_start)#, pred_goal_noise, pred_goal_start)



    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        pred_x_start_prev,
        t,
        t_prev,
        batch,
        if_prev=False
    ):
        

        # MODEL PRED
        preds = self.model_predictions(x=x,
                                       pred_x_start_prev=pred_x_start_prev,
                                       t=t,
                                       obs=batch['obs'] * batch['mask_past'],
                                       stage_masks=batch['mask_all'], gt_goal=batch['goal'],gt_goal_one_hot=batch['goal_one_hot'])
        pred_x_start = preds.pred_x_start
        pred_noise = preds.pred_noise

        #pred_goal_start = preds.pred_goal_start
        #pred_goal_noise = preds.pred_goal_noise

        # PRED X_0
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Compute mean and var
        noise = torch.randn_like(x) 
        mean_pred = (
                pred_x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
        )

        nonzero_mask = (1 - (t == 0).float()).reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))


        # # PRED goal
        # alpha_bar_goal = extract(self.alphas_cumprod, t, gt_goal_one_hot.shape)
        # if if_prev:
        #     alpha_bar_prev_goal = extract(self.alphas_cumprod_prev, t_prev, gt_goal_one_hot.shape)
        # else:
        #     alpha_bar_prev_goal = extract(self.alphas_cumprod, t_prev, gt_goal_one_hot.shape)
        # sigma_goal = (
        #         self.eta
        #         * torch.sqrt((1 - alpha_bar_prev_goal) / (1 - alpha_bar_goal))
        #         * torch.sqrt(1 - alpha_bar_goal / alpha_bar_prev_goal)
        # )

        # # Compute mean and var
        # noise_goal = torch.randn_like(x) 
        # mean_pred_goal = (
        #         pred_goal_start * torch.sqrt(alpha_bar_prev_goal)
        #         + torch.sqrt(1 - alpha_bar_prev_goal - sigma_goal ** 2) * pred_goal_noise
        # )

        #nonzero_mask_goal = (1 - (t == 0).float()).reshape(gt_goal_one_hot.shape[0], *((1,) * (len(gt_goal_one_hot.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * noise, pred_x_start#, mean_pred_goal+nonzero_mask_goal*sigma_goal*noise_goal, pred_goal_start

   

    @torch.no_grad()
    def p_sample_loop_with_input(
        self,
        batch,
        init_rand=None,
        n_diffusion_steps=-1,
    ):
        
        # INPUT
        device = self.betas.device
        x_0_pred = torch.zeros_like(batch["x_0"]).to(batch["x_0"].device)  # only used for shape

    
        # INIT PREDICTION (normal distr noise)
        pred = torch.randn_like(x_0_pred, device=device) if init_rand is None else init_rand  # BS x T x C
        init_noise = pred.clone()
        pred = pred.contiguous()


        # SAMPLE
        assert n_diffusion_steps == len(self.ddim_timestep_seq) # 50
        
        # Resample (DDIM)
        for t in tqdm(
            reversed(range(0, n_diffusion_steps)),
            desc="Resampled sampling loop time step",
            total=n_diffusion_steps,
            position=0,
            leave=True
        ):

            batched_times = torch.full((pred.shape[0],), self.ddim_timestep_seq[t], device=pred.device, dtype=torch.long)
            if t == 0:
                batched_times_prev = torch.full((pred.shape[0],), 0, device=device, dtype=torch.long)
                pred, x_0_pred = self.p_sample_ddim(
                    x=pred, 
                    pred_x_start_prev=x_0_pred,
                    t=batched_times, 
                    t_prev=batched_times_prev, 
                    batch=batch,
                    if_prev=True
                )
            else:
                batched_times_prev = torch.full((pred.shape[0],), self.ddim_timestep_seq[t-1], device=device, dtype=torch.long)
                pred, x_0_pred = self.p_sample_ddim(
                    x=pred,
                    pred_x_start_prev=x_0_pred, 
                    t=batched_times,
                    t_prev=batched_times_prev,
                    batch=batch
                )
        return pred, init_noise



    ''' Actual inference step '''
    def predict(
        self,
        x_0,
        obs,
        mask_past,
        masks_stages,
        *,
        n_samples=2,
        return_noise=False,
        n_diffusion_steps=-1,
        goal=None,gt_goal_one_hot=None,
    ):
        
        # Initialize observation
        
        obs = repeat(obs, "b t c -> (s b) t c", s=n_samples)
        x_0 = repeat(x_0, "b t c -> (s b) t c ", s=n_samples)
        mask_past = repeat(mask_past, "b t 1 -> (s b) t c", s=n_samples, c=obs.shape[-1])
        masks_stages = [repeat(mask.to(torch.bool), "b t c -> (s b) t c", s=n_samples) for mask in masks_stages]
        goal = repeat(goal, "b c -> (s b) c", s=n_samples)
        gt_goal_one_hot = repeat(gt_goal_one_hot, "b t c -> (s b) t c", s=n_samples)
        # Sample from the diffusion model
        
        x_out, init_noise = self.p_sample_loop_with_input(
            batch={
                "x_0": x_0,  # only used for shape
                "obs": obs,
                "mask_past": mask_past.to(torch.bool),
                "mask_all": masks_stages,
                "goal": goal,
                "goal_one_hot":gt_goal_one_hot,
            },
            init_rand=None,
            n_diffusion_steps=n_diffusion_steps
        )
          
        # Return
        init_noise = init_noise[0]
        if return_noise:
            assert n_samples == 1
            return x_out, init_noise
        return rearrange(x_out, "(s b) t c -> s b c t", s=n_samples)


