# main.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import common
from experiments import run_all_once_yield

# pho/rho 默认 1R
RHO   = "1R"   # 可用 '0.5R' / '2R' / '4R' / 数值
KAPPA = 4
SEED  = 2025

HEFT_EXTRA_COMM_S = 0.04
ENABLE_CROSS_COMM = True
ENABLE_INTRA_COMM = True

def main():
    seg, edg = common.load_segments_edges()

    rows = []
    for row in run_all_once_yield(
        seg, edg,
        rho=RHO, kappa=int(KAPPA),
        seed=SEED,
        heft_extra_comm_s=HEFT_EXTRA_COMM_S,
        enable_cross_comm=ENABLE_CROSS_COMM,
        enable_intra_comm=ENABLE_INTRA_COMM
    ):
        # 每个算法结束就打印；rho 显示为 'xR'
        print(f"rho={row['rho']}, kappa={row['kappa']}, {row['method']}={row['makespan']:.3f}")
        rows.append(row)

    out = Path("./output/data"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["rho","kappa","method","makespan"]].to_csv(out/"sweep.csv", index=False)
    print(f"Saved -> {out/'sweep.csv'}")

if __name__ == "__main__":
    main()
