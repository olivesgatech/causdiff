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

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    Qwen2VLForConditionalGeneration = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

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
    if m.__class__.__name__.endswith("Qwen2VLModel"):
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
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        # LoRA
        enable_lora: bool = True,
        lora_rank: int = 2,
        lora_alpha: int = 4,
        lora_dropout: float = 0.05,
        target_modules=("q_proj", "k_proj"),
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
            model_id, trust_remote_code=True, cache_dir=cache_dir
        )

        load_kwargs = dict(trust_remote_code=True, cache_dir=cache_dir)
        if device == "cuda":
            load_kwargs.update(dict(device_map="auto", torch_dtype=torch_dtype))
        else:
            load_kwargs.update(dict(torch_dtype=torch.float32))
        if attn_impl is not None:
            load_kwargs.update(dict(attn_implementation=attn_impl))

        # Load model
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
        except Exception:
            assert Qwen2VLForConditionalGeneration is not None, \
                "transformers>=4.43가 필요합니다 (Qwen2VLForConditionalGeneration)."
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

        # Optional: gradient checkpointing
        if enable_grad_ckpt and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Attach LoRA
        if enable_lora:
            assert get_peft_model is not None, "peft가 설치되어 있어야 합니다."
            peft_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(target_modules),
                bias="none",
                task_type="CAUSAL_LM"
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

        # Freeze vision tower / projector if requested
        if freeze_vision_tower:
            for name in ("vision_tower", "visual"):
                vt = getattr(self.model, name, None)
                if vt is not None:
                    for p in vt.parameters():
                        p.requires_grad = False

        if freeze_mm_projector:
            for name in ("multi_modal_projector", "mm_projector"):
                mm = getattr(self.model, name, None)
                if mm is not None:
                    for p in mm.parameters():
                        p.requires_grad = False

    

    def build_messages(self, global_intention, frame_idx, total_len, prev_subintents=None, recent_k=3):
        recent = prev_subintents[-recent_k:] if prev_subintents else []

        system_txt = (
            "You are a precise vision-language annotator for capturing fine-grained human intentions.\n"
            "Respect temporal logic using the frame index and visible evidence only. \n"
            "A fine-grained sub-intention is an atomic, observable, and immediate human intention (1–3s) with one main verb, one primary object/target.\n"
            "Avoid repeating recent intention as possible."
            "You must not generate something that is not in the image."
        )

        user_txt = (
            f"This video is for achieving one global intention: {global_intention}.\n"
            f"Frame index: {frame_idx} out of {total_len}. Recent actions' intentions: {recent}.\n"
            f"Based on what you can see in the image, provide diverse and novel intention as possible.\n"
            "Since you need to find immediate fine-grained intention, it is better if you observe where human moves towards to and where human is looking at."
            "Focus on the actually visible hand–object contact, actually visible tool usage, and small, precise manipulations or state changes.\n"
            "Think about action_verb, actually visible object, target_location, and what is the intention about that action_verb, and with that, deduct the fine grained intention."
            "Return the deducted fine-grained intention. (word max <= 8 words)"
        )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_txt}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_txt},
                {"type": "image", "image": "<image-placeholder>"}
            ]}
        ]

    # -------------------------
    # (Optional) simple resize
    # -------------------------
    @staticmethod
    def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
        if max(img.size) <= max_side:
            return img
        w, h = img.size
        s = max_side / float(max(w, h))
        new_w, new_h = int(w * s), int(h * s)
        return img.resize((new_w, new_h), Image.BICUBIC)

    def _decode_phrase(self, msgs_img, img, max_new_tokens=10):
        # msgs_img: apply_chat_template용으로 이미지 바인딩까지 끝난 메시지
        prompt = self.processor.apply_chat_template(msgs_img, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():  # phrase만 얻고 그래프는 만들지 않음 (LoRA 학습은 latent 경로에서만)
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        out = self.processor.batch_decode(
            gen_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]
        phrase = out.strip().split("\n")[0].strip()
        # 아주 가볍게 후처리
        phrase = phrase.rstrip(".")
        return phrase[:64]  # 과도한 길이 방지

    def forward(
        self,
        images: List[List[Image.Image]],        # (B, T)
        global_intention: Optional[str] = "Making pancake",
        frame_indices: Optional[List[List[int]]] = None,
        total_len: Optional[List[int]] = None,
        recent_subs_batch: Optional[List[List[str]]] = None,
        return_all_tokens: bool = False, chunk_size=4, recent_k=3,
    ) -> torch.Tensor:
        """
        Returns:
            vlm_latents : torch.Tensor, shape (B, T, D)
        """
        self.model.train()  # LoRA 학습을 위해 train 모드 (dropout 없음: LoRA만 학습)
        device = self.device

        B = len(images)
        assert B > 0 and isinstance(images[0], (list, tuple)), "images must be (B, T) list of lists"
        T = len(images[0])

        # 브로드캐스팅
        if isinstance(global_intention, list):
            assert len(global_intention) == B, "global_intention list length must match batch size"
            gi_list = global_intention
        else:
            gi_list = [global_intention] * B

        if total_len is None:
            total_list = [T] * B
        else:
            assert len(total_len) == B
            total_list = total_len

        running_recent = []
        for b in range(B):
            if recent_subs_batch and len(recent_subs_batch) > b and recent_subs_batch[b]:
                running_recent.append(list(recent_subs_batch[b]))
            else:
                running_recent.append([])

        running_recent = []
        for b in range(B):
            if recent_subs_batch and len(recent_subs_batch) > b and recent_subs_batch[b]:
                running_recent.append(list(recent_subs_batch[b]))
            else:
                running_recent.append([])

        pooled_B = []

        backbone = _unwrap_base_model(self.model)
        layers = _find_layers_module(backbone)
        norm_layer = _find_final_norm(backbone)

        for b in range(B):
            pooled_T = []
            for t0 in range(0, T, chunk_size):
                t1 = min(T, t0 + chunk_size)

                # --- hook 준비
                feat = {}
                hook = None
                if layers is not None:
                    last_layer = layers[-1]
                    def _save_from_last_layer(mod, finput, foutput):
                        feat["last"] = foutput[0] if isinstance(foutput, (tuple, list)) else foutput
                    hook = last_layer.register_forward_hook(_save_from_last_layer)
                elif norm_layer is not None:
                    def _save_from_norm(mod, finput, foutput):
                        feat["last"] = foutput
                    hook = norm_layer.register_forward_hook(_save_from_norm)
                else:
                    raise RuntimeError("Could not find decoder layers or final norm to hook.")

                # --- 청크 단위로 입력 구성
                img_chunk = images[b][t0:t1]
                msgs_chunk = []
                for dt, img in enumerate(img_chunk):
                    fi = (frame_indices[b][t0+dt] if frame_indices is not None else (t0+dt))
                    tl = (total_len[b] if total_len is not None else T)
                    msgs = self.build_messages(
                        global_intention=gi_list[b],
                        frame_idx=fi,
                        total_len=tl,
                        prev_subintents=running_recent[b][-recent_k:]
                    )
                    
                    # 실제 이미지 바인딩
                    msg_img = []
                    for turn in msgs:
                        content = []
                        for c in turn["content"]:
                            content.append({"type":"image","image":img} if c.get("type")=="image" else c)
                        msg_img.append({"role":turn["role"], "content":content})
                    # Qwen 입력으로 변환
                    prompt = self.processor.apply_chat_template(msg_img, add_generation_prompt=False)
                    inputs = self.processor(text=prompt, images=[img], return_tensors="pt")
                    msgs_chunk.append(inputs)

                # --- 배치로 병합 (동형 텐서라면 cat / 아니면 순차 수행)
                # 안전하게 순차 수행(메모리 우선)
                for inputs in msgs_chunk:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    # forward (학습 그래프 유지)
                    outputs = self.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)

                    if "last" not in feat:
                        # 일부 구현에서 hook이 레이어 호출 후에만 채워지므로, 다시 체크
                        if "last" not in feat:
                            raise RuntimeError("Failed to capture final hidden states via hook.")

                    last = feat.pop("last")        # (1, S, D)

                    if "attention_mask" in inputs:
                        mask = inputs["attention_mask"].float().unsqueeze(-1)
                        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    else:
                        pooled = last.mean(dim=1)

                    pooled_T.append(pooled)  # (1, D)

                    phrase = self._decode_phrase(msg_img, img, max_new_tokens=10)
                    if phrase:
                        if (len(running_recent[b]) == 0) or (phrase != running_recent[b][-1]):
                            running_recent[b].append(phrase)
                            # 지나치게 길어지지 않게 관리(필요시)
                            if len(running_recent[b]) > 30:
                                running_recent[b] = running_recent[b][-30:]

                    # 이 청크 step의 임시 변수 즉시 정리 (그래프는 pooled가 참조하므로 유지)
                    del outputs, inputs, last
                    torch.cuda.empty_cache()

                if hook is not None:
                    hook.remove()
            pooled_T = torch.cat(pooled_T, dim=0)   # (T, D) or (sum_S, D)
            pooled_B.append(pooled_T)  # (1, T, D)

        vlm_latents = pad_sequence(pooled_B, batch_first=True, padding_value=0.0)  # (B, T_max, D)
        
        return vlm_latents
