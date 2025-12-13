# run_3_hetero.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield  # 复用统一入口（支持 vgpu_weights）

# ===== 基础设置 =====
SEG_PATH = "../../input/segments_base.csv"
SEG_HEAVY_PATH = "../../input/segments_heavy.csv"
EDG_PATH = "../../input/edges.csv"

RHO = "1R"  # 只改变异构度；需要多 rho 可改为列表并外层再套一层循环
KAPPA = 4  # 题意固定 kappa=4
SEED = 2025
H_LIST = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,0.90,1.0]

OUT_DIR = Path("../../output/data")
OUT_FILE_NAME = "e3.csv"
OUT_LONGTAIL_FILE_NAME = "e3_longtail.csv"

UNEQUAL_WEIGHTS = {
    0.00: [0.25, 0.25, 0.25, 0.25],
    0.10: [0.28, 0.27, 0.23, 0.22],
    0.20: [0.31, 0.29, 0.21, 0.19],
    0.30: [0.34, 0.30, 0.20, 0.16],
    0.40: [0.37, 0.32, 0.18, 0.13],
    0.50: [0.40, 0.34, 0.16, 0.10],
    0.60: [0.50, 0.23, 0.15, 0.12],
    0.70: [0.55, 0.20, 0.15, 0.10],
    0.80: [0.60, 0.20, 0.10, 0.10],
    0.90: [0.64, 0.15, 0.11, 0.10],
    1.00: [0.70, 0.10, 0.10, 0.10],
}


# def weights_for_H_k4(H: float) -> list[float]:
#     """
#     为 kappa=4 构造一组权重 w，使 H = std(w)/mean(w) = H。
#     取 w = [1+H, 1+H, 1-H, 1-H]（均值=1, std=H；H<1 时均为正）。
#     experiments/common 内部会把它们归一化成每台机的 S_G_k。
#     """
#     if not (0.0 <= H < 1.0):
#         raise ValueError(f"H 必须在 [0,1) 内，收到 {H}")
#     return [1.0 + H, 1.0 + H, 1.0 - H, 1.0 - H]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)
    rows: list[dict] = []
    process(seg, edg, rows)
    # 写盘：H,method,makespan
    pd.DataFrame(rows, columns=["H", "method", "makespan"]).to_csv(OUT_DIR / OUT_FILE_NAME, index=False)
    print(f"Saved -> {OUT_DIR / OUT_FILE_NAME}")

    seg, edg = common.load_segments_edges(SEG_HEAVY_PATH, EDG_PATH)
    rows: list[dict] = []
    process(seg, edg, rows)
    # 写盘：H,method,makespan
    pd.DataFrame(rows, columns=["H", "method", "makespan"]).to_csv(OUT_DIR / OUT_LONGTAIL_FILE_NAME, index=False)
    print(f"Saved -> {OUT_DIR / OUT_LONGTAIL_FILE_NAME}")


def process(seg, edg, rows):
    for H in H_LIST:
        # w = weights_for_H_k4(H)
        w = UNEQUAL_WEIGHTS.get(H)
        # 复用统一入口；唯一差别：传 vgpu_weights（用来生成不均等 vGPU 切分）
        for row in run_all_once_yield(
                seg, edg,
                rho=RHO,
                kappa=int(KAPPA),
                seed=SEED,
                vgpu_weights=w,
        ):
            # 只保留三列：H, method, makespan
            rows.append({
                "H": float(H),
                "method": row["method"],
                "makespan": float(row["makespan"]),
            })
            print(f"H={H:.2f}, {row['method']}={row['makespan']:.3f}")


if __name__ == "__main__":
    main()
