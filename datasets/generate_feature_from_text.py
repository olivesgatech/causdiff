#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default='/home/hice1/skim3513/scratch/causdiff/datasets/darai/out.txt',
                    help="out.txt 경로 (줄마다 한 문장)")
    ap.add_argument("--outdir", dest="out_dir", type=Path, default='/home/hice1/skim3513/scratch/causdiff/datasets/darai/features_description_text',
                    help="임베딩을 저장할 폴더 (없으면 생성)")
    ap.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                    help="CLIP 텍스트 인코더 모델명 (512차원)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="L2 정규화를 끕니다(기본은 정규화).")
    args = ap.parse_args()

    texts = []
    with args.in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = " ".join(line.strip().split())
            if t:
                texts.append(t)

    if not texts:
        print("[WARN] 입력 문장이 없습니다.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_fp = (args.out_dir / "index.jsonl").open("w", encoding="utf-8")

    with torch.inference_mode():
        for i, t in enumerate(texts):
            inputs = tokenizer([t], padding=True, truncation=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**inputs)  # (1, D)
            if not args.no_normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.detach().cpu().numpy().astype("float32")  # (1,512) for ViT-B/32

            out_path = args.out_dir / f"{i:05d}.npy"
            np.save(out_path, arr)

            rec = {"id": i, "text": t, "path": str(out_path)}
            index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    index_fp.close()
    print(f"[OK] {len(texts)}개 문장 임베딩 완료. 저장 폴더: {args.out_dir}")

if __name__ == "__main__":
    main()
