# run_1_2_unequal.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield  # 复用同一个入口，只多传 vgpu_weights

# RHO_CHOICES = ["0.5R", "0.7R","0.9R", "1.0R", "1.1R","1.3R","1.5R"]
# KAPPA_CHOICES = [1, 2, 4, 7]
RHO_CHOICES = ["0.5R", "0.6R","0.75R", "0.9R", "1.0R", "1.25R","1.5R","2.0R"]  # 也可换成具体数字或混用
KAPPA_CHOICES = [4]
SEED = 2025
HEFT_EXTRA_COMM_S = 0.04
ENABLE_CROSS_COMM = True
ENABLE_INTRA_COMM = True

# 为每个 κ 指定不均等 vGPU 权重（长度必须等于 κ；不要求和为 1，会在 common 内归一化）
UNEQUAL_WEIGHTS = {
    1: [1.0],
    2: [0.3, 0.7],
    # 3: [0.2,0.3,0.5],
    4: [0.1, 0.2, 0.3, 0.4],
    # 5: [0.1, 0.1, 0.2, 0.2, 0.4],
    # 6: [0.1, 0.1, 0.1, 0.2, 0.2, 0.3],
    7: [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
    # 8: [0.03, 0.07, 0.10, 0.10, 0.10, 0.10, 0.20, 0.30],
}

SEG_PATH = "../../input/segments_base.csv"
SEG_HEAVY_PATH = "../../input/segments_heavy.csv"
EDG_PATH = "../../input/edges.csv"

OUT_PATH = "../../output/data"
OUT_FILE_NAME = "e1_unequal.csv"
OUT_LONGTAIL_FILE_NAME = "e1_unequal_longtail.csv"

# OUT_DIR = Path("../../output/data")
# OUT_CSV = OUT_DIR / "e1_unequal.csv"


def main():
    # OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(OUT_PATH)
    out.mkdir(parents=True, exist_ok=True)
    # # 首次写表头（四列）
    # if not OUT_CSV.exists():
    #     pd.DataFrame(columns=["rho", "kappa", "method", "makespan"]).to_csv(OUT_CSV, index=False)

    # 基础数据集
    rows = []
    seg, edg = common.load_segments_edges(SEG_PATH,EDG_PATH)
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / OUT_FILE_NAME, index=False)
    print(f"Saved -> {out / OUT_FILE_NAME}")

    # 长尾数据集处理
    rows = []
    seg, edg = common.load_segments_edges(SEG_HEAVY_PATH, EDG_PATH)
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / OUT_LONGTAIL_FILE_NAME, index=False)
    print(f"Saved -> {out / OUT_LONGTAIL_FILE_NAME}")


def process(edg, seg, rows):
    for rho in RHO_CHOICES:
        for kappa in KAPPA_CHOICES:
            weights = UNEQUAL_WEIGHTS.get(kappa)
            if not weights:
                continue

            # 传 vgpu_weights 代表这是异构vGPU
            for row in run_all_once_yield(
                    seg, edg,
                    rho=rho,kappa=int(kappa),
                    seed=SEED,
                    vgpu_weights=weights,  # 异构vGPU权重
                    heft_extra_comm_s=HEFT_EXTRA_COMM_S,
                    enable_cross_comm=ENABLE_CROSS_COMM,
                    enable_intra_comm=ENABLE_INTRA_COMM
            ):
                print(f"rho={row['rho']},  kappa={row['kappa']},  {row['method']}={row['makespan']:.3f}")
                rows.append(row)

if __name__ == "__main__":
    main()
