# run_once_unequal.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

import common  # 复用：common.run_all_once_yield
import experiments

# ===== 你想单次跑哪一组，就改这里 =====
RHO   = "1R"   # 可填 "0.5R" / "1R" / "2R" / "4R" / 数字 / "auto"
KAPPA = 4      # vGPU 数
SEED  = 2025

# 为这次单次运行指定不均等 vGPU 权重（长度必须等于 KAPPA；不要求和为 1）
VGPU_WEIGHTS = {
    1: [1.0],
    2: [0.3, 0.7],
    4: [0.1, 0.2, 0.3, 0.4],
    8: [0.03, 0.07, 0.10, 0.10, 0.10, 0.10, 0.20, 0.30],
}.get(KAPPA, None)

# 读写路径
SEG_PATH = "./input/segments_base.csv"
EDG_PATH = "./input/edges.csv"
OUT_DIR  = Path("./output/data")
OUT_CSV  = OUT_DIR / "e1_unequal_once.csv"


def main():
    # 基础准备
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)

    # 单次运行：调用 common.run_all_once_yield（和 run_equal 完全一致的入口）
    rows = []
    for row in experiments.run_all_once_yield(
        seg, edg,
        rho=RHO,
        kappa=int(KAPPA),
        seed=SEED,
        vgpu_weights=VGPU_WEIGHTS,   # 仅此一处与 run_equal 不同：传入不均等权重
        # 其余参数走默认值：heft_extra_comm_s / enable_cross_comm / enable_intra_comm
    ):
        print(f"rho={row['rho']}, kappa={row['kappa']}, {row['method']}={row['makespan']:.6f}")
        rows.append(row)

    # 输出到 CSV（覆盖写，避免重复累计），与 run_equal 一致的四列
    # pd.DataFrame(rows, columns=["rho", "kappa", "method", "makespan"]).to_csv(OUT_CSV, index=False)
    # print(f"[saved] {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
