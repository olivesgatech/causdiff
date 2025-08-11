#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import argparse

def normalize(s: str) -> str:
    # 공백 정규화 + 앞뒤 공백 제거
    return " ".join(str(s).split()).strip()

def extract_justif(in_path: Path, out_path: Path) -> None:
    seen = set()         # 중복 체크(소문자 기준)
    uniques = []

    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            just = None
            if isinstance(obj, dict):
                res = obj.get("result") or {}
                if isinstance(res, dict):
                    just = res.get("justif_brief") or res.get("justification")

            if not just:
                continue

            j = normalize(just)
            if not j:
                continue

            key = j.lower()
            if key in seen:
                continue
            seen.add(key)
            uniques.append(j)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for j in uniques:
            w.write(j + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path("/home/hice1/skim3513/scratch/causdiff/datasets/darai/out.jsonl"),
                    help="입력 JSONL 파일 경로")
    ap.add_argument("--out", dest="out_path", type=Path,
                    default=Path("/home/hice1/skim3513/scratch/causdiff/datasets/darai/out.txt"),
                    help="출력 TXT 파일 경로")
    args = ap.parse_args()
    extract_justif(args.in_path, args.out_path)

if __name__ == "__main__":
    main()
