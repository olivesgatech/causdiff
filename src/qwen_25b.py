# qwen_online_encoder.py
# ------------------------------------------------------------
# Qwen2-VL Online Encoder with LoRA (backbone pooling only)
# - Safe PEFT unwrap → call Qwen2VLModel directly (not CausalLM)
# - Memory-friendly: no generate(), no all hidden states dump
# - Outputs pooled pre-generation latents (B, T, D)
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, AutoModelForVision2Seq
import contextlib

from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model


def _unwrap_base_model(m):
    # PeftModel이면 base로 내려가고, 거기서 다시 .model 이 있으면 그걸 백본으로
    base = getattr(m, "get_base_model", None)
    if callable(base):
        m = m.get_base_model()
    # Qwen2-VL ForConditionalGeneration는 보통 .model 안에 백본이 들어있음
    return getattr(m, "model", m)

def _find_layers_module(backbone):
    # 가장 흔한 경로: backbone.layers (ModuleList)
    if hasattr(backbone, "layers") and isinstance(backbone.layers, torch.nn.ModuleList) and len(backbone.layers) > 0:
        return backbone.layers
    # 다른 경로들도 시도 (안전하게 순회)
    for name, mod in backbone.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            return mod
    return None  # 못 찾으면 None

def _find_final_norm(backbone):
    # llama류는 보통 model.norm / norm
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(backbone, attr):
            return getattr(backbone, attr)
    # 백업: 이름에 'norm'이 들어가는 마지막 모듈
    cand = None
    for name, mod in backbone.named_modules():
        if "norm" in name.lower():
            cand = mod
    return cand     # 없으면 None


def _get_qwen_backbone(model: nn.Module) -> nn.Module:
    """
    Any-wrapping → return Qwen2VLModel (backbone)
    Works with PeftModel(Qwen2VLForConditionalGeneration) and plain models.
    """
    m = model

    # 1) PEFT unwrap
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model"):
        try:
            m = m.base_model
        except Exception:
            pass

    # 2) CausalLM wrapper → .model
    if hasattr(m, "model") and m.__class__.__name__.endswith("ForConditionalGeneration"):
        return m.model

    # 3) already backbone
    if m.__class__.__name__.endswith("Qwen2_5_VLModel"):
        return m

    # 4) fallback common names
    for name in ("language_model", "transformer", "backbone"):
        if hasattr(m, name):
            return getattr(m, name)

    raise RuntimeError(f"[QwenVLOnlineEncoder] Could not locate Qwen2VLModel backbone from type: {type(model)}")


class QwenVLOnlineEncoder(nn.Module):
    """
    비디오 프레임(이미지 시퀀스)을 받아 Qwen2-VL의 pre-generation latent를 뽑아내는 엔코더.

    입력
    ----
    images : List[List[PIL.Image]]  (B, T)
        - 각 시퀀스는 T장의 이미지 (예: 두 뷰를 수평 concat한 1장도 OK)
    global_intention : str 또는 List[str]
    frame_indices : Optional[List[List[int]]]  (B, T)
    total_len : Optional[List[int]]            (B,)
    recent_subs_batch : Optional[List[List[str]]]  (B, variable length)

    출력
    ----
    vlm_latents : torch.Tensor, shape (B, T, D) (보통 D = 4096 for Qwen2-VL-7B)

    특징
    ----
    - generate() 사용 안 하고 backbone(last_hidden_state)만 풀링
    - LoRA 삽입(modules: q/k/v/o proj). 비전 타워/멀티모달 projector는 기본 동결
    - grad는 LoRA 파트로 흘러가며, 메모리/속도 효율적
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        # LoRA
        enable_lora: bool = True,
        lora_rank: int = 2,
        lora_alpha: int = 4,
        lora_dropout: float = 0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        # Freeze options
        freeze_vision_tower: bool = True,
        freeze_mm_projector: bool = True,
        # Misc
        max_side: int = 336,
        enable_grad_ckpt: bool = False,
        attn_impl: Optional[str] = None,  # e.g., "flash_attention_2" if supported
        
    ):
        super().__init__()
        self.device = device
        self.max_side = max_side

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,  min_pixels=224*224, max_pixels=1024*1024,
        )

        load_kwargs = dict(trust_remote_code=True, cache_dir=cache_dir, torch_dtype=torch.bfloat16)
        
        if attn_impl is not None:
            load_kwargs.update(dict(attn_implementation=attn_impl))

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)

        if enable_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Attach LoRA
        if enable_lora:
            self.finetune = True
            assert get_peft_model is not None, "peft가 설치되어 있어야 합니다."
            peft_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(target_modules),
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )
            self.model = get_peft_model(self.model, peft_cfg)

            # 1) 전체 동결
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            K = 6  # 마지막 6블록만
            layers = getattr(self.model, "model", None)
            layers = getattr(layers, "layers", None) or []
            for li, blk in enumerate(layers):
                enable = (li >= len(layers) - K)
                for name, module in blk.named_modules():
                    if "lora_" in name:
                        for p in module.parameters():
                            p.requires_grad = enable
        else:
            self.finetune = False

        # Freeze vision tower / projector if requested
        if not freeze_vision_tower:
            for name in ("vision_tower", "visual"):
                vt = getattr(self.model, name, None)
                if vt is not None:
                    for p in vt.parameters():
                        p.requires_grad = True

        if not freeze_mm_projector:
            for name in ("multi_modal_projector", "mm_projector"):
                mm = getattr(self.model, name, None)
                if mm is not None:
                    for p in mm.parameters():
                        p.requires_grad = True

    def forward(
        self,
        images,                         # List[List[PIL.Image]]  (B, variable T_b)
        global_intention="Making pancake",
        frame_indices=None,
        total_len=None,
        recent_subs_batch=None,
        return_all_tokens=False,
        chunk_size=16,
        recent_k=0,
        use_autocast=True,
    ):
        if self.finetune == False:
            self.model.eval()
        else:
            self.model.train()
        device = self.device

        B = len(images)
        assert B > 0 and isinstance(images[0], (list, tuple))

        # broadcast
        if isinstance(global_intention, list):
            assert len(global_intention) == B
            gi_list = global_intention
        else:
            gi_list = [global_intention] * B
        if total_len is None:
            total_list = [len(images[b]) for b in range(B)]
        else:
            assert len(total_len) == B
            total_list = total_len

        # 각 시퀀스 길이 수집
        lengths = [len(seq) for seq in images]          # e.g., [61, 60, 61, ...]
        assert all(l > 0 for l in lengths)

        # (B, T_b) → flat
        flat_imgs, flat_prompts = [], []
        for b in range(B):
            T_b = lengths[b]
            for t in range(T_b):
                img = images[b][t]
                if max(img.size) > self.max_side:
                    w, h = img.size
                    scale = self.max_side / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))
                flat_imgs.append(img)
                flat_prompts.append(
                    f"Global intention: {gi_list[b]}. Frame {t}/{total_list[b]}. "
                    "Do not repeat the Global intention."
                )

        # 미니배치로 인코딩
        #backbone = _get_qwen_backbone(self.model)
        pooled_list = []
        ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
        with ctx:
            for i0 in range(0, len(flat_imgs), chunk_size):
                i1 = min(len(flat_imgs), i0 + chunk_size)
                msgs_batch = []
                imgs_batch = []
                for j in range(i0, i1):
                    img = flat_imgs[j]
                    txt = flat_prompts[j]

                    # Qwen2-VL 권장 포맷: 메시지에 이미지 placeholder 포함
                    msgs = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a intention tagger. Output fine-grained intention based only on evidence from the given image. Definition: fine-grained = next 1-5s sub-step"}]},
                        {"role": "user", "content": [
                            {"type": "image", "image": "<image-placeholder>"},
                            {"type": "text", "text": txt}
                        ]}
                    ]
                    msgs_batch.append(self.processor.apply_chat_template(
                        msgs, add_generation_prompt=False
                    ))
                    imgs_batch.append(img)
                inputs = self.processor(
                    text=msgs_batch,
                    images=imgs_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                for k in inputs:
                    inputs[k] = inputs[k].to(device, non_blocking=True)

                #outputs = backbone(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,          # ✅ 마지막 히든만 씀
                    use_cache=False,
                    return_dict=True
                )
                last = outputs.hidden_states[-1]                  # (mb, S, D)
                #last = outputs.last_hidden_state
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"].float().unsqueeze(-1)
                    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                else:
                    pooled = last.mean(dim=1)
                pooled_list.append(pooled)                        # (mb, D)

        # (N, D)
        vlm_flat = torch.cat(pooled_list, dim=0)

        # 다시 (B, T_b, D)로 분해 → pad
        chunks = []
        idx = 0
        for b in range(B):
            T_b = lengths[b]
            chunks.append(vlm_flat[idx:idx+T_b])                  # (T_b, D)
            idx += T_b
        # pad to (B, T_max, D)
        vlm_latents = pad_sequence(chunks, batch_first=True, padding_value=0.0)

        return vlm_latents    # (B, T_max, D)

    # QwenVLOnlineEncoder 내부에 추가
    @torch.no_grad()
    def generate_phrases(
        self,
        images,                         # List[List[PIL.Image]]  (B, variable T_b)
        global_intention="Making pancake",
        frame_indices=None,
        total_len=None,
        # 디코딩/품질 옵션
        max_new_tokens=1200,              # <= 8 words 보장용 여유 토큰
        do_sample=False,                # 재현성 우선이면 False, 다양성 원하면 True + top_p/temperature
        top_p=0.9,
        temperature=0.7,
        num_beams=1,                    # 안정성 원하면 2~3
        repetition_penalty=1.1,        # 경미한 반복 억제
        chunk_size=16,
        max_length_tokens=128,          # 입력 토큰 상한
        enforce_8_words=True,           # 후처리 강제
        remove_punct=True,              # 후처리 구두점 제거
        recent_k=0,                     # 필요시 최근 생성문장 정보 프롬프트 주입
        use_autocast=True
    ):
        """
        학습 때 쓰던 '이미지 placeholder + 동일 시스템/유저 메시지' 프롬프트로
        파인튜닝된 Qwen2-VL(+LoRA)에서 프레임별 짧은 의도 문장을 생성한다.
        반환: List[List[str]]  (B, T_b)
        """
        # 0) eval 모드 + LoRA 활성
        self.model.eval()

        device = self.device
        B = len(images)
        assert B > 0 and isinstance(images[0], (list, tuple))

        # broadcast
        if isinstance(global_intention, list):
            assert len(global_intention) == B
            gi_list = global_intention
        else:
            gi_list = [global_intention] * B

        if total_len is None:
            total_list = [len(seq) for seq in images]
        else:
            assert len(total_len) == B
            total_list = total_len

        lengths = [len(seq) for seq in images]
        assert all(l > 0 for l in lengths)

        # 1) 플랫 전개 + 동일 프롬프트 구성 (학습 포맷 유지)
        flat_imgs, flat_msgs = [], []
        for b in range(B):
            T_b = lengths[b]
            for t in range(T_b):
                img = images[b][t]
                if max(img.size) > self.max_side:
                    w, h = img.size
                    scale = self.max_side / float(max(w, h))
                    img = img.resize((int(w * scale), int(h * scale)))

                
                # 시스템/유저 프롬프트: 학습 때와 동일한 메시지 스타일
                msgs = [
                    {"role": "system", "content": [
                        {"type": "text", "text": "Output a fine-grained intention (>=20 words) based only on visible evidence."}
                    ]},
                    {"role": "user", "content": [
                        {"type": "image", "image": "<image-placeholder>"},
                        {"type": "text", "text":
                            f"Global intention: {gi_list[b]}. Frame {t}/{total_list[b]}. "
                        }
                    ]}
                ]
                flat_imgs.append(img)
                flat_msgs.append(self.processor.apply_chat_template(
                    msgs, add_generation_prompt=True
                ))

        # 2) 청크 단위 디코딩
        outs_flat = []
        ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()
        with ctx:
            for i0 in range(0, len(flat_imgs), chunk_size):
                i1 = min(len(flat_imgs), i0 + chunk_size)
                inputs = self.processor(
                    text=flat_msgs[i0:i1],
                    images=flat_imgs[i0:i1],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
                dec_list = self.processor.batch_decode(
                    gen_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                for s in dec_list:
                    # 1) 모든 줄을 공백으로 이어붙이기
                    #    - strip해서 빈 줄 제거, ' '로 join
                    parts = [line.strip() for line in s.replace("\r", "\n").split("\n") if line.strip()]
                    joined = " ".join(parts) if parts else ""

                    # 2) 공백 정규화 (여러 공백 -> 하나)
                    joined = " ".join(joined.split())

                    # 3) 아주 가벼운 후처리만 (원하면 최소화)
                    if remove_punct and joined.endswith("."):
                        joined = joined[:-1]

                    outs_flat.append(joined)

        # 3) (B, T_b)로 재조립 + 간단 smoothing/anti-repetition
        outs_bt, idx = [], 0
        for b in range(B):
            T_b = lengths[b]
            seq = outs_flat[idx:idx+T_b]
            idx += T_b

            # 간단 anti-repetition: 바로 이전과 동일하면 마지막 토큰 제거
            for t in range(1, T_b):
                if seq[t] == seq[t-1]:
                    ws = seq[t].split()
                    if len(ws) > 2:
                        seq[t] = " ".join(ws[:-1])
            outs_bt.append(seq)

        return outs_bt

