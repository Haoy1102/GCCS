# run_1_rho.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import common
from experiments import run_all_once_yield

SEED = 2025
HEFT_EXTRA_COMM_S = 0.04
ENABLE_CROSS_COMM = True
ENABLE_INTRA_COMM = True
RHO_CHOICES = ["0.5R", "0.7R","0.9R", "1.0R", "1.1R","1.3R","1.5R"]  # 也可换成具体数字或混用
KAPPA_CHOICES = [1, 2, 4, 7]

def main():

    out = Path("../../output/data");
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    seg, edg = common.load_segments_edges()
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / "e1_equal.csv", index=False)
    print(f"Saved -> {out / 'e1_equal.csv'}")

    # 长尾数据集处理
    seg, edg = common.load_segments_edges("./input/segments_heavy.csv", "./input/edges.csv")
    rows = []
    process(edg, seg, rows)
    pd.DataFrame(rows)[["rho", "kappa", "method", "makespan"]].to_csv(out / "e1_equal_longtail.csv", index=False)
    print(f"Saved -> {out / 'e1_equal_longtail.csv'}")


def process(edg, seg, rows):

    for rho in RHO_CHOICES:
        for kappa in KAPPA_CHOICES:
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


if __name__ == "__main__":
    main()
