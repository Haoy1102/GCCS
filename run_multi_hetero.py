# run_multi_hetero.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

import common
from experiments import run_all_once_yield   # 复用统一入口（支持 vgpu_weights）

# ===== 基础设置 =====
SEG_PATH = "./input/segments_base.csv"
EDG_PATH = "./input/edges.csv"

RHO       = "1R"         # 只改变异构度；需要多 rho 可改为列表并外层再套一层循环
KAPPA     = 4            # 题意固定 kappa=4
SEED      = 2025
H_LIST    = [0.00, 0.10, 0.20, 0.30, 0.40]   # 想多测就加

OUT_DIR   = Path("./output/data")
OUT_CSV   = OUT_DIR / "e2_heterogeneity.csv"


def weights_for_H_k4(H: float) -> list[float]:
    """
    为 kappa=4 构造一组权重 w，使 H = std(w)/mean(w) = H。
    取 w = [1+H, 1+H, 1-H, 1-H]（均值=1, std=H；H<1 时均为正）。
    experiments/common 内部会把它们归一化成每台机的 S_G_k。
    """
    if not (0.0 <= H < 1.0):
        raise ValueError(f"H 必须在 [0,1) 内，收到 {H}")
    return [1.0 + H, 1.0 + H, 1.0 - H, 1.0 - H]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)

    rows: list[dict] = []

    for H in H_LIST:
        w = weights_for_H_k4(H)

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

    # 写盘：H,method,makespan
    pd.DataFrame(rows, columns=["H", "method", "makespan"]).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
