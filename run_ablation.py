# run_ablation.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

import common
from algo import GCCS as algo_gccs
from experiments import _resolve_rho, _format_rho_as_R

# 固定一次的参数（与其它 run_* 脚本对齐）
RHO   = "1R"
KAPPA = 4
SEED  = 2025

SEG_PATH = "./input/segments_base.csv"
EDG_PATH = "./input/edges.csv"

OUT_DIR = Path("./output/data")
OUT_CSV = OUT_DIR / "e4_ablation.csv"

# 四个变体（与题述 a/b/c/d 一一对应）
VARIANTS = [
    ("normal",      {"phase1_mode": "lp",      "gpu_queue_mode": "dynamic"}),  # 正常
    ("no-pack",     {"phase1_mode": "random",  "gpu_queue_mode": "dynamic"}),  # 去掉 min-max 打包
    ("no-dynamic",  {"phase1_mode": "lp",      "gpu_queue_mode": "fixed"}),    # 去掉动态 vGPU
    ("no-all",        {"phase1_mode": "random",  "gpu_queue_mode": "fixed"}),    # 两个都去掉
    # 想用“全局独立选择”替代随机，把 "random" 改成 "global" 即可
]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)

    rows = []

    rho_val = _resolve_rho("auto", seg)
    rho_str = _format_rho_as_R(rho_val, seg)
    cluster = common.make_default_cluster(
        num_servers=6, rho=rho_val, kappa=int(KAPPA),
        segments=seg, seed=SEED
    )

    for name, cfg in VARIANTS:
        # 只改“第一阶段打包策略 + vGPU 分配策略”，其余流程保持不动
        ms, _ = algo_gccs.run(
            seg, edg,
            cluster=cluster,            # 让内部按 RHO/KAPPA 自构集群（如果你 run 里本就这样）
            seed=SEED,
            phase1_mode=cfg["phase1_mode"],
            gpu_queue_mode=cfg["gpu_queue_mode"],
        )
        print(f"[E4] {name}: makespan={ms:.3f}")
        rows.append({"variant": name, "makespan": float(ms)})

    pd.DataFrame(rows, columns=["variant", "makespan"]).to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
