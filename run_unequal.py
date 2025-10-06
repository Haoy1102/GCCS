# run_unequal.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield  # 复用同一个入口，只多传 vgpu_weights

# ------------- 实验网格（与 run_equal 风格一致） -------------
RHO_CHOICES   = ["0.5R", "1R", "2R", "4R"]
KAPPA_CHOICES = [1, 2, 4, 8]
SEED = 2025

# 为每个 κ 指定不均等 vGPU 权重（长度必须等于 κ；不要求和为 1，会在 common 内归一化）
UNEQUAL_WEIGHTS = {
    1: [1.0],
    2: [0.3, 0.7],
    4: [0.1, 0.2, 0.3, 0.4],
    8: [0.03, 0.07, 0.10, 0.10, 0.10, 0.10, 0.20, 0.30],
}

# 输出
OUT_DIR = Path("./output/data")
OUT_CSV = OUT_DIR / "e1_unequal.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg, edg = common.load_segments_edges("./input/segments_base.csv", "./input/edges.csv")

    # 首次写表头（四列）
    if not OUT_CSV.exists():
        pd.DataFrame(columns=["rho", "kappa", "method", "makespan"]).to_csv(OUT_CSV, index=False)

    rows = []
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
                vgpu_weights=weights,           # <<< 不均等就在这里生效
                # 下面三个参数保持默认（你 experiments 里会在 HEFT/MRSA 内处理）
                # heft_extra_comm_s=0.04,
                # enable_cross_comm=True,
                # enable_intra_comm=True,
            ):
                print(f"rho={row['rho']},  kappa={row['kappa']},  {row['method']}={row['makespan']:.3f}")
                rows.append(row)

    # 统一写盘（四列）
    # pd.DataFrame(rows, columns=["rho", "kappa", "method", "makespan"]) \
    #   .to_csv(OUT_CSV, mode="a", header=False, index=False)
    # print(f"[saved] {OUT_CSV.resolve()}")
    out = Path("./output/data"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["rho","kappa","method","makespan"]].to_csv(out/"e1_unequal.csv", index=False)
    print(f"Saved -> {out/'e1_unequal.csv'}")


if __name__ == "__main__":
    main()
