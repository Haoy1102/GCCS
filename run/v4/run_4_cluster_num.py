# run_1_rho.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import common
from experiments import run_all_once_yield

SEED = 2025
RHO_CHOICES = ["1.0R"]  # 也可换成具体数字或混用
KAPPA_CHOICES = [4]

SEG_PATH = "../../input/segments_base.csv"
SEG_HEAVY_PATH = "../../input/segments_heavy.csv"
EDG_PATH = "../../input/edges.csv"

OUT_PATH = "../../output/data"
OUT_FILE_NAME = "e4_1_equal.csv"
# OUT_LONGTAIL_FILE_NAME = "e1_2_equal_longtail.csv"
OUT_HERE_FILE_NAME = "e4_2_unequal.csv"
# OUT_HERE_LONGTAIL_FILE_NAME = "e1_4_unequal_longtail.csv"

CLUSTER_NUMS = [4,6,8,12,16]
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

def main():
    out = Path(OUT_PATH)
    out.mkdir(parents=True, exist_ok=True)

    # 同构标准数据集处理
    rows = []
    seg, edg = common.load_segments_edges(SEG_PATH,EDG_PATH)
    process(edg, seg, rows,False)
    pd.DataFrame(rows)[["cluster_num", "rho", "kappa", "method", "makespan"]].to_csv(out / OUT_FILE_NAME, index=False)
    print(f"Saved -> {out / OUT_FILE_NAME}")

    # 异构标准数据集处理
    rows = []
    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)
    process(edg, seg, rows, True)
    pd.DataFrame(rows)[["cluster_num", "rho", "kappa", "method", "makespan"]].to_csv(out / OUT_HERE_FILE_NAME, index=False)
    print(f"Saved -> {out / OUT_HERE_FILE_NAME}")


def process(edg, seg, rows, is_heter):
    for rho in RHO_CHOICES:
        for kappa in KAPPA_CHOICES:
            for cluster_num in CLUSTER_NUMS:
                weights = None
                if is_heter:
                    weights = UNEQUAL_WEIGHTS.get(kappa)
                    if not weights:
                        continue

                for row in run_all_once_yield(
                        seg, edg,
                        rho=rho, kappa=int(kappa),
                        seed=SEED,
                        vgpu_weights=weights,  # 异构vGPU权重
                        cluster_num=cluster_num,
                ):
                    rows.append({
                        "cluster_num": cluster_num,
                        "rho": rho,
                        "kappa": int(kappa),
                        "method": row["method"],
                        "makespan": float(row["makespan"]),
                    })
                    print(f"cluster_num={cluster_num}, rho={row['rho']}, kappa={row['kappa']}, {row['method']}={row['makespan']:.3f}")


if __name__ == "__main__":
    main()
