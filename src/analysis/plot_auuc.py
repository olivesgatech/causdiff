#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_auuc_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "group": r["group"],                          # "20_group" or "30_group"
                "T_max": int(r["T_max"]),
                "start_idx": int(r["start_idx"]),
                "end_idx": int(r["end_idx"]),
                "used_len": int(r["used_len"]),
                "baseline_area": float(r["baseline_area"]),
                "proposed_area": float(r["proposed_area"]),
                "unit": r.get("unit", "nat·step"),
            })
    return rows

def build_series(rows, group_name):
    """
    특정 group("20_group"/"30_group")의 결과를 end_idx 오름차순으로 정렬.
    반환:
      x: prediction rates [%], 보통 [10,20,30,50]에서 길이 n(최대 4)
      y_b: baseline AUUC 길이 n
      y_p: proposed AUUC 길이 n
      unit: 문자열
    """
    filt = [r for r in rows if r["group"] == group_name]
    if len(filt) == 0:
        return np.array([]), np.array([]), np.array([]), "nat·step"
    filt = sorted(filt, key=lambda r: r["end_idx"])

    # 표준 레이트 (윈도우 4개 기준)
    std_rates = np.array([10, 20, 30, 50], dtype=float)
    n = min(len(filt), len(std_rates))
    x = std_rates[:n]
    y_b = np.array([filt[i]["baseline_area"] for i in range(n)], dtype=float)
    y_p = np.array([filt[i]["proposed_area"] for i in range(n)], dtype=float)
    unit = filt[0]["unit"]
    return x, y_b, y_p, unit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default='/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/auuc_windows.csv')
    ap.add_argument("--out_png", type=str, default="/home/hice1/skim3513/scratch/causdiff/outputs/nturgbd/auuc_rate_lines_semantics.png")
    ap.add_argument("--title", type=str, default="AUUC vs Prediction Rate (combined)")
    args = ap.parse_args()

    rows = load_auuc_csv(args.csv_path)

    # 시리즈 구성
    x20, y20_b, y20_p, unit20 = build_series(rows, "20_group")
    x30, y30_b, y30_p, unit30 = build_series(rows, "30_group")

    if x20.size == 0 and x30.size == 0:
        raise RuntimeError("CSV에 20_group/30_group 데이터가 없습니다.")

    # 색상 지정
    color_baseline_20 = "#76b7eb"  # 하늘색
    color_baseline_30 = "#1f77b4"  # 파란색
    color_proposed_20 = "#f7a6b8"  # 연분홍
    color_proposed_30 = "#d62728"  # 분홍

    ms = 6  # marker size

    plt.figure(figsize=(5, 2.0))

    plt.plot(x20, y20_b, "-o", color=color_baseline_20, linewidth=2, markersize=ms)
    plt.plot(x20, y20_p, "-o", color=color_proposed_20, linewidth=2, markersize=ms)

    plt.plot(x30, y30_b, "-o", color=color_baseline_30, linewidth=2, markersize=ms)
    plt.plot(x30, y30_p, "-o", color=color_proposed_30, linewidth=2, markersize=ms)

    # 축/레이블
    # y축 단위는 두 그룹이 같다고 가정(혼재라면 unit20 우선)
    y_unit = unit20 if x20.size > 0 else unit30
    plt.xticks([10, 20, 30, 50])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close()
    print(f"Saved figure -> {args.out_png}")

if __name__ == "__main__":
    main()
