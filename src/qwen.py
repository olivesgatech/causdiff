#!/usr/bin/env python3
"""
IntenDiff – Sub‑Intention Latent Extractor & Decoder (Qwen2‑VL)

What this script does
---------------------
1) Loads frames from a directory containing a single video worth of images.
   • Path: /home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd/Making pancake/camera_1_fps_15
   • Only files whose basename starts with "01_4" are processed (e.g., 01_4xxxx.jpg).
2) Iterates the video in coarse "scenes" (configurable stride), and for each selected frame:
   • Prompts a VLM (Qwen2‑VL) with the global intention "Making pancake" and the list
     of previously inferred sub‑intentions to get the current sub‑intention text.
   • Extracts a latent vector *before* text generation by pooling hidden states from
     the final transformer layer for the multimodal prompt (see notes below).
   • Saves the latent vector to disk as: {frame_basename}_{frame_index}.npy
   • Also saves a compact JSONL log with the generated sub‑intention text and metadata.
3) Optionally (best‑effort) extracts a separate visual feature vector directly from the
   vision tower if accessible; saves alongside the pooled latent for analysis.

Notes on the "latent vector"
----------------------------
There is no single canonical "pre‑generation latent" for an LLM+VLM. Here we provide two
robust, implementation‑portable alternatives:
  A) pooled_mm_hidden: Mean‑pool of the final hidden states across all non‑padding tokens
     (multimodal: text + image placeholder tokens) *before* sampling any new tokens.
  B) pooled_vision (optional): If the model exposes a vision tower, we mean‑pool its final
     spatial features for the input image. If not available, this falls back to None.
These are saved so you can later baseline or ablate which correlates better with behavior.

Environment
-----------
• Requires: transformers >= 4.43, accelerate, torch, pillow, numpy, tqdm
• Model:   Qwen/Qwen2-VL-7B-Instruct  (configurable via --model-id)
• GPU recommended. bfloat16 by default; adjust to float16 if needed.

Usage
-----
python inten_subintent_qwen2vl_extract_and_decode.py \
  --root "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd/Making pancake/camera_1_fps_15" \
  --prefix "01_4" \
  --out "./outputs/making_pancake_01_4" \
  --scene-stride 15 --max-frames 0

"""

import os
import re
import json
import glob
import math
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
try:
    from transformers import Qwen2VLForConditionalGeneration  # HF >= 4.43
except Exception:
    Qwen2VLForConditionalGeneration = None


# ----------------------------
# Utilities
# ----------------------------

def natural_key(s: str):
    """Natural sort key that splits digits from non‑digits (e.g., f2 < f10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def list_frames(root: str, prefix: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, f"{prefix}*{ext}")))
    paths = sorted(paths, key=natural_key)
    return paths


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def hstack_same_height(img1: Image.Image, img2: Image.Image, gap= 0, bg=(0,0,0)):
    # 두 이미지의 세로를 맞춰 가로로 붙입니다.
    h = max(img1.height, img2.height)
    # 비율 유지하며 리사이즈
    def resize_to_h(img, target_h):
        w = int(round(img.width * (target_h / img.height)))
        return img.resize((w, target_h), Image.BICUBIC)
    if img1.height != h: img1 = resize_to_h(img1, h)
    if img2.height != h: img2 = resize_to_h(img2, h)
    w = img1.width + gap + img2.width
    out = Image.new("RGB", (w, h), bg)
    out.paste(img1, (0, 0))
    if gap > 0:
        # 간격은 배경색으로 채워짐
        out.paste(img2, (img1.width + gap, 0))
    else:
        out.paste(img2, (img1.width, 0))
    return out

# ----------------------------
# Qwen2‑VL helpers
# ----------------------------

def build_messages(global_intention, prev_subintents, frame_name, frame_idx, total_len, recent_k=3):
    recent = prev_subintents[-recent_k:] if prev_subintents else []

    system_txt = (
        "You are a precise vision-language annotator for capturing fine-grained human intentions.\n"
        "Respect temporal logic using the frame index and visible evidence only. \n"
        "A fine-grained sub-intention is an atomic, observable, and immediate human intention (1–3s) with one main verb, one primary object/target, in order to do one more fine-grained main verb.\n"
        "You are given two synchonized views merged into one image. You must return a structured fine-grained intention with a final short phrase in the end."
        "You must not generate something that is not in the image."
    )

    user_txt = (
        f"This video is for achieving one global intention: {global_intention}.\n"
        f"Frame index: {frame_idx} out of {total_len}. Recent actions' intentions: {recent}.\n"
        f"Based on what you can see in the image, provide diverse and novel intention as possible.\n"
        "Since you need to find immediate fine-grained intention, it is better if you observe where human moves towards to and where human is looking at."
        "Focus on the actually visible hand–object contact, actually visible tool usage, and small, precise manipulations or state changes.\n"
        "Think about action_verb, actually visible object, target_location, and what is the intention about that action_verb, and with that, deduct the fine grained intention."
        "Return the deducted fine-grained intention."
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": system_txt}]},
        {"role": "user", "content": [
            {"type": "text", "text": user_txt},
            {"type": "image", "image": "<image-placeholder>"}
        ]}
    ]
def list_frame_pairs(root1: str, root2: str, prefix: str):
    f1 = list_frames(root1, prefix)
    f2 = list_frames(root2, prefix)
    n = min(len(f1), len(f2))
    if n == 0:
        raise FileNotFoundError("No overlapping frames between the two roots.")
    if len(f1) != len(f2):
        print(f"[warn] length mismatch: cam1={len(f1)} cam2={len(f2)} → using min={n}")
    return f1[:n], f2[:n]


def extract_hidden_pooled(outputs, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Pool final hidden states across non‑pad tokens.
    Returns tensor of shape (hidden_dim,) or None.
    """
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        return None
    last = outputs.hidden_states[-1]  # (B, S, D)
    if attention_mask is None:
        # Mean over sequence
        pooled = last.mean(dim=1).squeeze(0)
        return pooled.detach().cpu()
    else:
        mask = attention_mask.float().unsqueeze(-1)  # (B, S, 1)
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (summed / denom).squeeze(0)
        return pooled.detach().cpu()


def try_extract_vision_pooled(model, pixel_values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Best‑effort extraction of pooled vision features.
    Not all Qwen2‑VL variants expose a public vision tower. We try a few common hooks.
    Returns CPU tensor shape (vision_dim,) or None.
    """
    if pixel_values is None:
        return None
    try:
        # Common patterns observed in Qwen2‑VL repos
        vt = getattr(model, "vision_tower", None) or getattr(model, "visual", None)
        if vt is None:
            return None
        with torch.no_grad():
            vis_out = vt(pixel_values)  # expect (B, T, D) or (B, D, H, W) depending on impl
        if isinstance(vis_out, (list, tuple)):
            vis = vis_out[0]
        else:
            vis = vis_out
        if vis.dim() == 4:  # (B, C, H, W) → mean‑pool spatial
            vis = vis.mean(dim=[2, 3])
        elif vis.dim() == 3:  # (B, T, D) → mean over tokens
            vis = vis.mean(dim=1)
        elif vis.dim() == 2:
            pass  # already (B, D)
        else:
            return None
        return vis.squeeze(0).detach().cpu()
    except Exception:
        return None

def _extract_until_end(text: str) -> str:
    # <END> 이전만 취득
    if "<END>" in text:
        text = text.split("<END>", 1)[0]
    return text.strip()

def _extract_json_minified(s: str):
    # 한 줄 JSON 추출: 첫 { ... } 블록
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _kv_lines_to_json(s: str):
    """
    JSON이 아니고 'key: value' 형태(줄바꿈 또는 '\\n' 포함)일 때 최대한 보정.
    phrase가 비어있거나 키가 누락되면 null로 채움.
    """
    # 리터럴 \n 을 실제 개행으로 바꿔보기
    s = s.replace("\\n", "\n")
    out = {
        "phase_idx": None, "transition": "continue",
        "action_verb": None, "object": None, "tool_or_hand": None,
        "target_location": None, "phrase": None
    }
    for line in s.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in out:
            out[k] = v if v else None
    # 최소 보정: phrase가 없으면 동사+객체로 구성
    if not out["phrase"]:
        ph = []
        if out["action_verb"]: ph.append(out["action_verb"])
        if out["object"]: ph.append(out["object"])
        out["phrase"] = " ".join(ph)[:64] if ph else ""
    return out

def generate_subintention_text(model, processor, messages, image, device, max_new_tokens=1000):
    # 메시지 구성
    msg_with_img = []
    for turn in messages:
        new_turn = {"role": turn["role"], "content": []}
        for c in turn["content"]:
            if c.get("type") == "image":
                new_turn["content"].append({"type": "image", "image": image})
            else:
                new_turn["content"].append(c)
        msg_with_img.append(new_turn)

    prompt = processor.apply_chat_template(msg_with_img, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model(**inputs)
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # 한 줄 구조는 greedy가 보통 더 안정적
            num_beams=1,
            no_repeat_ngram_size=2,
            repetition_penalty=1.05,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    text = processor.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

    # <END> 기준 자르기 + 첫 줄만
    if "<END>" in text:
        text = text.split("<END>", 1)[0]
    phrase = text.splitlines()[0].strip()

    # 후처리: 길이 제한, 끝의 문장부호 제거
    phrase = " ".join(phrase.split())        # 공백 정규화
    if len(phrase.split()) > 100:
        phrase = " ".join(phrase.split()[:100])
    phrase = phrase.rstrip(".;:,-")
    return phrase



def forward_and_latents(model, processor, messages, image: Image.Image, device: str):
    """Run a *non‑generative* forward pass to capture hidden states before decoding."""
    msg_with_img = []
    for turn in messages:
        new_turn = {"role": turn["role"], "content": []}
        for c in turn["content"]:
            if c.get("type") == "image":
                new_turn["content"].append({"type": "image", "image": image})
            else:
                new_turn["content"].append(c)
        msg_with_img.append(new_turn)

    prompt = processor.apply_chat_template(msg_with_img, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    pooled_mm = extract_hidden_pooled(outputs, attention_mask=inputs.get("attention_mask"))

    # Try to also capture a pure visual pooled vector if possible
    pixel_values = inputs.get("pixel_values")
    pooled_vision = try_extract_vision_pooled(model, pixel_values)

    return pooled_mm, pooled_vision, inputs


# ----------------------------
# Main pipeline
# ----------------------------

def run(
    root: str,
    prefix: str,
    out_dir: str,
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    device_pref: str = "auto",
    dtype: str = "bfloat16",
    scene_stride: int = 15,
    max_frames: int = 0,
    global_intention: str = "Making pancake",
):
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "latents"))

    hf_cache = getattr(args, "hf_cache", "/home/hice1/skim3513/scratch/causdiff/hf_cache")
    ensure_dir(hf_cache)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache, "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", os.path.dirname(hf_cache))

    # 1) Collect frames
    if 'camera_1' in root:
        root2 = root.replace('camera_1', 'camera_2')
    else:
        root2 = root.replace('camera_2', 'camera_1')

    frames1, frames2 = list_frame_pairs(root, root2, prefix)
    # frames = list_frames(root, prefix)
    # if not frames:
    #     raise FileNotFoundError(f"No frames found with prefix '{prefix}' under {root}")

    # Subsample by scene_stride to approximate scene changes (configurable)
    indices = list(range(0, min(len(frames1), len(frames2)), max(1, scene_stride)))
    if max_frames and max_frames > 0:
        indices = indices[:max_frames]

    # 2) Load model & processor
    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else (
        torch.float16 if dtype == "float16" and torch.cuda.is_available() else torch.float32
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=hf_cache)
    load_kwargs = dict(
        trust_remote_code=True,
        cache_dir=hf_cache,
    )
    if device == "cuda":
        load_kwargs.update(dict(device_map="auto", torch_dtype=torch_dtype))
    else:
        load_kwargs.update(dict(torch_dtype=torch.float32))

    model = None
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
    except Exception as e1:
        if Qwen2VLForConditionalGeneration is not None:
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            except Exception as e2:
                raise RuntimeError(f"Failed to load Qwen2-VL model as Vision2Seq or Qwen2VLForConditionalGeneration. e1={e1} e2={e2}")
        else:
            raise RuntimeError(f"Failed to load Qwen2-VL model as Vision2Seq. {e1}")
    if device == "cpu":
        model.to(device)
    if device == "cpu":
        model.to(device)

    # 3) Iterate and extract
    prev_subintents: List[str] = []

    log_path = os.path.join(out_dir, "subintent_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as logf:
        for k in tqdm(indices, desc="Processing frames"):
            # fpath = frames[k]
            # fname = os.path.basename(fpath)
            fpath1 = frames1[k]; fpath2 = frames2[k]
            fname = os.path.basename(fpath1)
            frame_idx = k

            # Load image
            #img = Image.open(fpath).convert("RGB")
            img1 = Image.open(fpath1).convert("RGB")
            img2 = Image.open(fpath2).convert("RGB")

            #img = resize_to_max_pixels(img, max_pixels=args.max_pixels, max_side=args.max_side, size_factor=args.size_factor)
            img1 = resize_to_max_pixels(img1, max_pixels=args.max_pixels, max_side=args.max_side, size_factor=args.size_factor)
            img2 = resize_to_max_pixels(img2, max_pixels=args.max_pixels, max_side=args.max_side, size_factor=args.size_factor)
            img = hstack_same_height(img1, img2)
            img = resize_to_max_pixels(img, max_pixels=args.max_pixels*2, max_side=args.max_side*2, size_factor=args.size_factor)
            fpath = f"{fpath1} | {fpath2}"

            # Build messages
            messages = build_messages(global_intention, prev_subintents, fname, k, len(frames1))

            # (A) Non‑generative forward → pooled latents
            pooled_mm, pooled_vision, inputs = forward_and_latents(model, processor, messages, img, device)

            # (B) Text generation for verification/decoding
            subintent_text = generate_subintention_text(model, processor, messages, img, device)
            print(subintent_text)
            if not prev_subintents or subintent_text != prev_subintents[-1]:
                prev_subintents.append(subintent_text)

            # Save latents: {file_basename}_{frame_index}.npy
            base_noext, _ = os.path.splitext(fname)
            latent_basename = f"{base_noext}_{frame_idx:06d}"

            latents_to_save = {}
            if pooled_mm is not None:
                latents_to_save["pooled_mm_hidden"] = pooled_mm.numpy()
            if pooled_vision is not None:
                latents_to_save["pooled_vision"] = pooled_vision.numpy()

            if latents_to_save:
                # Pack into a single npz per frame to keep both vectors aligned
                npz_path = os.path.join(out_dir, "latents", f"{latent_basename}.npz")
                np.savez_compressed(npz_path, **latents_to_save)

            # Write JSONL log row
            row = {
                "frame_path": fpath,
                "frame_index": int(frame_idx),
                "latent_file": f"latents/{latent_basename}.npz" if latents_to_save else None,
                "sub_intention": subintent_text,
                "global_intention": global_intention,
                "prev_subintent_count": len(prev_subintents) - 1,
                "model_id": model_id,
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Saved latents under: {os.path.join(out_dir, 'latents')}\nLog: {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Extract sub‑intention latents and texts with Qwen2‑VL")
    p.add_argument("--root", type=str, 
                   help="Root directory containing frames (video split into images)")
    p.add_argument("--prefix", type=str, default="01_4",
                   help="Only process files whose basename starts with this prefix")
    p.add_argument("--out", type=str, 
                   help="Output directory for latents and logs")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                   help="HF Hub model id for Qwen2‑VL Instruct variant")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device preference. 'auto' picks CUDA if available")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Compute dtype (GPU recommended)")
    p.add_argument("--scene-stride", type=int, default=15,
                   help="Take one frame every N frames as a proxy for scene changes")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Optional upper limit on processed frames (0 = all per stride)")
    p.add_argument("--global-intention", type=str, default="Making pancake",
                   help="Global intention string used in prompts")
    p.add_argument("--hf-cache", type=str,
                   default="/home/hice1/skim3513/scratch/causdiff/hf_cache",
                   help="Hugging Face cache directory (models, tokenizer, processor will be downloaded here)")
    p.add_argument("--max-pixels", type=int, default=180000,
                   help="safe pixel number (H*W)")
    p.add_argument("--max-side", type=int, default=512,
               help="최장변 상한. 초과 시 다운스케일")
    p.add_argument("--size-factor", type=int, default=32,
                   help="resize (normally 14, 16, 32)")

    return p.parse_args()

def resize_to_max_pixels(img: Image.Image, max_pixels: int, max_side: int, size_factor: int = 32) -> Image.Image:
    w, h = img.size
    # 1) 최장변 캡
    scale_side = 1.0
    if max(w, h) > max_side:
        scale_side = max_side / float(max(w, h))

    # 2) 면적(픽셀수) 캡
    cur_pixels = w * h
    scale_area = 1.0
    if cur_pixels > max_pixels:
        scale_area = math.sqrt(max_pixels / float(cur_pixels))

    # 3) 가장 보수적인(더 작은) 스케일 선택
    scale = min(scale_side, scale_area)

    new_w = max(size_factor, int(w * scale) // size_factor * size_factor)
    new_h = max(size_factor, int(h * scale) // size_factor * size_factor)

    # 최종 안전장치
    new_w = max(size_factor, new_w)
    new_h = max(size_factor, new_h)

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img



if __name__ == "__main__":
    # ---- 설정 ----
    SPLIT_PATH = "/home/hice1/skim3513/scratch/causdiff/datasets/darai/splits/train_split.txt"  # 필요시 절대경로로 바꿔도 됨
    BASE_RGB_SD = "/home/hice1/skim3513/AIFirst_F24_data/darai/RGB_sd"

    args = parse_args()
    args.model_id = "Qwen/Qwen2-VL-7B-Instruct"
    args.scene_stride = 15
    args.max_frames = 0
    args.dtype = "bfloat16"
    args.device = "auto"
    if not getattr(args, "hf_cache", None):
        args.hf_cache = "/home/hice1/skim3513/scratch/causdiff/hf_cache"

    # 결과 폴더 구성 편의를 위해 의도명을 슬러그로 변환
    def slugify(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())

    # 파일명 포맷: 14_3_camera_1_fps_15_Making pancake.txt
    # 정규식으로 prefix, camera, intention 추출
    pattern = re.compile(r"^(\d+_\d+)_camera_(1|2)_fps_15_(.+)\.txt$")

    # train_split.txt 읽기
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for line in lines:
        m = pattern.match(line)
        if not m:
            print(f"[skip] Unrecognized line format: {line}")
            continue

        prefix, cam_id, intention = m.group(1), m.group(2), m.group(3)

        # camera_2 항목은 camera_1 처리로 커버되므로 스킵
        if cam_id == "2":
            print(f"[skip] camera_2 line (covered by camera_1 pairing): {line}")
            continue

        # camera_1만 처리
        root = os.path.join(BASE_RGB_SD, intention, "camera_1_fps_15")
        out_dir = os.path.join(
            "./outputs",
            f"{slugify(intention)}",
            f"{prefix}_qwen7b_twoview"
        )
        os.makedirs(out_dir, exist_ok=True)

        # args 세팅
        args.root = root
        args.prefix = prefix
        args.out = out_dir
        # 의도 문자열을 run()에 넘겨주기 위해 args에 넣어둠
        args.global_intention = intention

        print("\n" + "=" * 80)
        print(f"[RUN] prefix={prefix} | intention='{intention}' | root={root}")
        print(f"[OUT] {out_dir}")
        print("=" * 80)

        try:
            run(
                root=args.root,
                prefix=args.prefix,
                out_dir=args.out,
                model_id=args.model_id,
                device_pref=args.device,
                dtype=args.dtype,
                scene_stride=args.scene_stride,
                max_frames=args.max_frames,
                global_intention=args.global_intention,
            )
        except Exception as e:
            print(f"[error] Failed on line: {line}\n{e}")
            # 실패해도 다음 라인 계속
            continue
