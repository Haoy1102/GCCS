# sweep_main.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import common
from experiments import run_all_once_yield

def main():
    seg, edg = common.load_segments_edges()
    R = common.workload_ratio_R(seg)

    rho_list = ["0.5R", "1R", "2R", "4R"]  # 也可换成具体数字或混用
    kappas   = [2, 3, 4, 5, 6]
    SEED     = 2025

    HEFT_EXTRA_COMM_S = 0.04
    ENABLE_CROSS_COMM = True
    ENABLE_INTRA_COMM = True

    rows = []
    for rho in rho_list:
        for kappa in kappas:
            for row in run_all_once_yield(
                seg, edg,
                rho=rho, kappa=int(kappa),
                seed=SEED,
                heft_extra_comm_s=HEFT_EXTRA_COMM_S,
                enable_cross_comm=ENABLE_CROSS_COMM,
                enable_intra_comm=ENABLE_INTRA_COMM
            ):
                # rho 已是 'xR' 字符串
                print(f"rho={row['rho']}, kappa={row['kappa']}, {row['method']}={row['makespan']:.3f}")
                rows.append(row)

    out = Path("./output/data"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["rho","kappa","method","makespan"]].to_csv(out/"base_ex.csv", index=False)
    print(f"Saved -> {out/'base_ex.csv'}")

if __name__ == "__main__":
    main()
