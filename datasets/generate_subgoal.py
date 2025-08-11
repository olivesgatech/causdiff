from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re, json
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
from PIL import Image
def parse_gt(gt_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parse ground truth file with lines like:
    /abs/path/to/20_3_00000.jpg, UNDEFINED, UNDEFINED
    Returns: { '20_3_00000.jpg': ('UNDEFINED','UNDEFINED'), ... }
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    with gt_file.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Robust split: first token is path, rest two are labels
            parts = [p.strip() for p in s.split(',')]
            if len(parts) < 3:
                # Try another split strategy (comma + space)
                parts = [p.strip() for p in s.split(', ')]
            if len(parts) < 3:
                # skip malformed
                continue
            img_path_raw = parts[0]
            l2 = parts[-2]
            l3 = parts[-1]
            basename = Path(img_path_raw).name
            mapping[basename] = (l2, l3)
    return mapping

def sorted_frames(img_dir: Path) -> List[str]:
    """
    Return basenames sorted lexicographically (safe due to zero-padding).
    """
    jpgs = sorted([p.name for p in img_dir.glob('*.jpg')])
    return jpgs

def get_prev_labels(current_basename: str, order: List[str], mapping: Dict[str, Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Collect L2/L3 labels for all frames strictly before current_basename in the given order.
    If a frame is missing from mapping, it will be skipped.
    """
    prev_l2, prev_l3 = [], []
    try:
        idx = order.index(current_basename)
    except ValueError:
        raise SystemExit(f"[Error] current frame '{current_basename}' not found in image directory listing.")
    for b in order[:idx]:
        if b in mapping:
            l2, l3 = mapping[b]
            prev_l2.append(l2)
            prev_l3.append(l3)
    return prev_l2, prev_l3

def build_prompt(l1: str,
                 prev_l2: List[str],
                 prev_l3: List[str],
                 curr_frame: str,
                 curr_l2: str,
                 curr_l3: str,
                 include_image: bool = False,
                 image_abs_path: Optional[str] = None) -> str:
    """
    Stronger prompt: forbid parroting L1/L2/L3, force image-grounded observation,
    and require purpose-phrased NEXT intention with allowed verbs.
    """
    print(curr_l2, curr_l3)
    # Normalize ban list (라벨 그대로/유사 표현 금지)
    ban_raw = [
        l1,
        curr_l2, curr_l3,
        "Making pancake", "Make pancake", "Make pancakes", "Making pancakes"
    ]
    ban = [b.strip() for b in ban_raw if b and b.strip() and b.upper() != "UNDEFINED"]
    ban_lower = sorted(set([b.lower() for b in ban]))

    prev_l2_str = ', '.join([x for x in prev_l2[-12:] if x and x.upper() != "UNDEFINED"]) or 'None'
    prev_l3_str = ', '.join([x for x in prev_l3[-12:] if x and x.upper() != "UNDEFINED"]) or 'None'

    vision_hint = ""
    if include_image and image_abs_path:
        vision_hint = (
            f"\nCURRENT_FRAME_IMAGE: {image_abs_path}\n"
            "(Use only what is visually observable: tools, surfaces, materials, hands, states.)\n"
        )

    allowed_verbs = (
        "heat to, grease to, pour to, spread to, whisk to, "
        "mix to, flip to, rest to, plate to, garnish to"
    )

    prompt = f"""### ROLE
You are an image-grounded intention planner. You must invent the immediate NEXT sub-intention
from the image, not copy labels.

### HARD RULES
1) Look at the image first. Describe (internally) what is happening based on visible cues.
2) Output ONE purpose-phrased subgoal using these verbs when possible: {allowed_verbs}.
3) The subgoal MUST NOT equal or paraphrase any of the following strings. If it does, rephrase to a finer, next-step intention:
BAN_LIST(lowercased): {ban_lower}
4) The subgoal must be local in time (next 3–10 seconds), not the overall task.
5) Prefer tool/object-anchored intentions (pan, batter, bowl, whisk, spatula, butter, oil).
6) Do NOT output just the global goal or the current L2/L3; create a new intention inferred from the image.

### NEGATIVE
- Bad: "{l1}"     (global goal)
- Bad: "{curr_l2}" (current L2 label)
- Bad: "{curr_l3}" (current L3 label)

### CONTEXT
GLOBAL GOAL (L1): "{l1}"

HISTORY (most recent first, up to 12):
- L2: {prev_l2_str}
- L3: {prev_l3_str}

CURRENT FRAME: {curr_frame}
- L2_now: {curr_l2}
- L3_now: {curr_l3}{vision_hint}

### OUTPUT FORMAT (JSON ONLY)
Return ONLY a JSON code block. The subgoal must NOT repeat items in BAN_LIST.

```json
{{
"subgoal": "<purpose phrasing>",
"justif_brief": "<=20 words about visible cues>"
}}
"""
    return prompt

def call_llm(prompt: str,
             mode: str = "vision",
             image_path: Optional[str] = None) -> Dict[str, str]:
    """
    Qwen/Qwen2-VL-2B-Instruct (HF, 로컬)
    - mode='vision' & image_path 제공 시 이미지+텍스트
    - mode='text'이면 텍스트만
    반환: {"subgoal": "...", "justif_brief": "..."}
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # --- 유틸 ---
    def _decode_generated_only(gen_ids, input_ids, tokenizer):
        gen_only = gen_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(gen_only, skip_special_tokens=True)

    def _extract_json(s: str) -> dict:
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 균형 괄호로 첫 JSON 객체만 추출
        start = s.find("{")
        while start != -1:
            stack = 0
            for i in range(start, len(s)):
                if s[i] == "{": stack += 1
                elif s[i] == "}":
                    stack -= 1
                    if stack == 0:
                        chunk = s[start:i+1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            break
            start = s.find("{", start+1)
        return {"subgoal": "unknown", "justif_brief": "JSON parse error"}

    # --- 모델/프로세서 로드(캐시) ---
    global _QWEN2VL_VLM, _QWEN2VL_PROC
    try:
        _QWEN2VL_VLM; _QWEN2VL_PROC
    except NameError:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _QWEN2VL_VLM = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            # trust_remote_code=True,  # 필요시 해제
        )
        _QWEN2VL_PROC = AutoProcessor.from_pretrained(model_id)

    model = _QWEN2VL_VLM
    processor = _QWEN2VL_PROC
    tok = processor.tokenizer

    # --- 메시지 구성 ---
    sys_msg = (
        "You are a strict JSON generator. "
        "Output ONLY a JSON code block with keys exactly: subgoal, justif_brief. "
        "No prose, no explanations."
    )
    user_content = [{"type": "text", "text": prompt + "\n\nReturn ONLY:\n```json\n{\n  \"subgoal\": \"...\",\n  \"justif_brief\": \"...\"\n}\n```\n"}]

    pil_img = None
    if mode == "vision" and image_path:
        pil_img = Image.open(image_path).convert("RGB")
        user_content.append({"type": "image", "image": pil_img})

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content},
    ]

    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # --- 인코딩 ---
    if pil_img is not None:
        inputs = processor(text=[chat_text], images=[pil_img], return_tensors="pt")
    else:
        inputs = processor(text=[chat_text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # --- 생성 ---
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    # --- 생성분만 디코드 & JSON 파싱 ---
    out_str = _decode_generated_only(gen, inputs["input_ids"], tok)
    payload = _extract_json(out_str)

    subgoal = payload.get("subgoal", "unknown")
    justif = payload.get("justif_brief", payload.get("justification", ""))[:200]
    return {"subgoal": subgoal, "justif_brief": justif}



def main(current):
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', type=Path, default='/home/hice1/skim3513/scratch/causdiff/datasets/darai/20_3_making_pancake')
    ap.add_argument('--gt_file', type=Path, default="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/darai/groundTruth_img/20_3_camera_2_fps_15_Making pancake.txt")
    ap.add_argument('--l1', type=str, default="Making Pancake")
    ap.add_argument('--include_image', action='store_true', default='true', help="Include current frame image path in the prompt")
    ap.add_argument('--mode', type=str, default='vision', choices=['text', 'vision'])
    ap.add_argument('--out', type=Path, default='/home/hice1/skim3513/scratch/causdiff/datasets/darai/out.jsonl', help="Path to save prompts/results as JSONL")
    args = ap.parse_args()

    args.current = current

    if not args.img_dir.exists():
        raise SystemExit(f"[Error] img_dir not found: {args.img_dir}")
    if not args.gt_file.exists():
        raise SystemExit(f"[Error] gt_file not found: {args.gt_file}")

    mapping = parse_gt(args.gt_file)
    order = sorted_frames(args.img_dir)
    if not order:
        raise SystemExit(f"[Error] no images found in {args.img_dir}")

    if args.current not in order:
        # Try to tolerate when user passes absolute path; coerce to basename
        args.current = Path(args.current).name
        if args.current not in order:
            raise SystemExit(f"[Error] current frame '{args.current}' not in image dir list.")

    # Current labels from mapping
    if args.current not in mapping:
        print(f"[Warn] current frame '{args.current}' not found in ground truth mapping; using UNDEFINED.", file=sys.stderr)
        curr_l2, curr_l3 = ("UNDEFINED", "UNDEFINED")
    else:
        curr_l2, curr_l3 = mapping[args.current]

    prev_l2, prev_l3 = get_prev_labels(args.current, order, mapping)

    # Build prompt
    image_abs_path = str(args.img_dir.joinpath(args.current).resolve()) if args.include_image else None
    prompt = build_prompt(args.l1, prev_l2, prev_l3, args.current, curr_l2, curr_l3,
                          include_image=args.include_image, image_abs_path=image_abs_path)

    # Output handling
    out_fp = args.out.open('a', encoding='utf-8') if args.out else None
    try:
        result = call_llm(prompt, mode='vision', image_path=image_abs_path)
        rec = {"frame": args.current, "prompt": prompt, "result": result}
        if out_fp:
            out_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
        else:
            print(json.dumps(rec, ensure_ascii=False, indent=2))
    finally:
        if out_fp:
            out_fp.close()

if __name__ == "__main__":
    idx = 0
    for jpg in os.listdir('/home/hice1/skim3513/scratch/causdiff/datasets/darai/20_3_making_pancake'):
        current = f"20_3_{idx:05d}.jpg"
        main(current)
        idx += 1