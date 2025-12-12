# run_1_2_unequal.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield  # 复用同一个入口，只多传 vgpu_weights

# ------------- 实验网格（与 run_equal 风格一致） -------------
RHO_CHOICES = ["0.5R", "0.7R","0.9R", "1.0R", "1.1R","1.3R","1.5R"]
KAPPA_CHOICES = [1, 2, 4, 7]
SEED = 2025

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

# 输出
OUT_DIR = Path("../../output/data")
OUT_CSV = OUT_DIR / "e1_unequal.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path("../../output/data");
    out.mkdir(parents=True, exist_ok=True)
    # 首次写表头（四列）
    if not OUT_CSV.exists():
        pd.DataFrame(columns=["rho", "kappa", "method", "makespan"]).to_csv(OUT_CSV, index=False)

    # 基础数据集
    rows = []
    seg, edg = common.load_segments_edges()
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / "e1_unequal.csv", index=False)
    print(f"Saved -> {out / 'e1_unequal.csv'}")

    # 长尾数据集处理
    rows = []
    seg, edg = common.load_segments_edges("./input/segments_heavy.csv", "./input/edges.csv")
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / "e1_unequal_longtail.csv", index=False)
    print(f"Saved -> {out / 'e1_unequal_longtail.csv'}")


def process(edg, seg, rows):
    for rho in RHO_CHOICES:
        for kappa in KAPPA_CHOICES:
            weights = UNEQUAL_WEIGHTS.get(kappa)
            if not weights:
                continue

            # 调 experiments 里的统一函数；唯一差别：传 vgpu_weights
            for row in run_all_once_yield(
                    seg, edg,
                    rho=rho,
                    kappa=int(kappa),
                    seed=SEED,
                    vgpu_weights=weights,  # <<< 不均等就在这里生效
                    # 下面三个参数保持默认（你 experiments 里会在 HEFT/MRSA 内处理）
                    # heft_extra_comm_s=0.04,
                    # enable_cross_comm=True,
                    # enable_intra_comm=True,
            ):
                print(f"rho={row['rho']},  kappa={row['kappa']},  {row['method']}={row['makespan']:.3f}")
                rows.append(row)

if __name__ == "__main__":
    main()
