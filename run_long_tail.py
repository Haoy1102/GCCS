# run_once_equal.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield

# pho/rho 与 kappa 固定不变
RHO   = "1R"
KAPPA = 4
SEED  = 2025

# HEFT/Hydra 通信参数保持你现有默认
HEFT_EXTRA_COMM_S = 0.04
ENABLE_CROSS_COMM = True
ENABLE_INTRA_COMM = True

# 两个数据集：Base 与 LongTail
DATASETS = [
    ("Balanced", "./input/segments_base.csv"),
    ("LongTail", "./input/segments_heavy.csv"),
]
EDG_PATH = "./input/edges.csv"

OUT_DIR = Path("./output/data")
OUT_CSV = OUT_DIR / "e3_longtail.csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_name, seg_path in DATASETS:
        seg, edg = common.load_segments_edges(seg_path, EDG_PATH)

        for row in run_all_once_yield(
            seg, edg,
            rho=RHO, kappa=int(KAPPA),
            seed=SEED,
            heft_extra_comm_s=HEFT_EXTRA_COMM_S,
            enable_cross_comm=ENABLE_CROSS_COMM,
            enable_intra_comm=ENABLE_INTRA_COMM
        ):
            # —— 即时打印 —— #
            print(f"[{ds_name}] rho={row['rho']}, kappa={row['kappa']}, {row['method']}={row['makespan']:.3f}")
            # —— 仅保留三列：dataset, method, makespan —— #
            rows.append({
                "dataset": ds_name,
                "method": row["method"],
                "makespan": float(row["makespan"]),
            })

    # 写出三列表
    pd.DataFrame(rows, columns=["dataset", "method", "makespan"]).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
