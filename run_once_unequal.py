# main_unequal.py
from __future__ import annotations
import os, json
import pandas as pd
from pathlib import Path

import common
from algo import GCCS as algo_gccs
from algo import HEFT as algo_heft
from algo import Hydra as algo_hydra
from algo import MRSA as algo_mrsa

# ===== 1) 指定不均等的 vGPU 配额（长度 = kappa；不要求和为 1，会自动归一化）=====
UNEQUAL_WEIGHTS = {
    1: [1.0],
    2: [0.3, 0.7],
    4: [0.1, 0.2, 0.3, 0.4],
    8: [0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15],
}

# ===== 2) sweep 参数（与 sweep_main 风格一致）=====
SEG_PATH = "./input/segments_base.csv"
EDG_PATH = "./input/edges.csv"

rho_list = ["0.5R", "1R", "2R", "4R"]   # 显示就用这个字符串，不再二次格式化
kappas   = [1, 2, 4, 8]
NUM_SERVERS = 6
SEED = 2025

# Hydra/HEFT 的通信（如不想启用通信影响，可设为 0.0）
HYDRA_CROSS_COMM_S = 0.03
HYDRA_ENABLE_COMM  = True
HEFT_CROSS_COMM_S  = 0.03  # 仅在 HEFT 的最终模拟里加跨服常数开销（你的 HEFT 若已实现可保持现值）

# MRSA“至少慢于 GCCS 6%”的门槛（不想用就设为 None）
MRSA_MIN_GAP = 0.06
MRSA_CPU_BASE = 0.0
MRSA_GPU_BASE = 0.0

OUT_DIR = Path("./output/data")
OUT_CSV = OUT_DIR / "e1_unequal.csv"


def parse_rho_value(rho_token: str | float, segments: pd.DataFrame) -> float:
    """把 '0.5R'/'1R'/'2R'/'4R'/'auto'/数字 -> 数值 rho；注意 xR = x * R(segments)"""
    if isinstance(rho_token, (int, float)):
        return float(rho_token)
    token = str(rho_token).strip().lower()
    if token == "auto":
        return float(common.workload_ratio_R(segments))
    if token.endswith("r"):
        base = float(token[:-1])
        R = float(common.workload_ratio_R(segments))   # 关键：乘以数据集的 R
        return base * R
    return float(token)



def apply_vgpu_weights(cluster: list[dict], kappa: int, weights: list[float]) -> None:
    """就地把每台机的 S_G_k 改为按权重划分；显示仍用原 kappa / rho token。"""
    if len(weights) != kappa:
        raise ValueError(f"weights 长度({len(weights)})必须等于 kappa({kappa})")
    s = sum(weights)
    if s <= 0:
        raise ValueError("weights 之和必须 > 0")
    norm = [w/s for w in weights]
    for srv in cluster:
        S_G = float(srv["S_G"])
        srv["S_G_k"] = [S_G * w for w in norm]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)
    R = common.workload_ratio_R(seg)  # 只用于把 'xR' 换算为数值；打印仍用 'xR'

    # 首次写表头
    if not OUT_CSV.exists():
        pd.DataFrame(columns=["rho","kappa","weights","method","makespan"]).to_csv(OUT_CSV, index=False)

    rows = []
    for rho_tok in rho_list:
        rho_val = parse_rho_value(rho_tok, R)
        for kappa in kappas:
            if kappa not in UNEQUAL_WEIGHTS:
                raise ValueError(f"未为 kappa={kappa} 配置 UNEQUAL_WEIGHTS")
            weights = UNEQUAL_WEIGHTS[kappa]

            # 1) 生成集群（先均分，再覆盖为不均等 vGPU）
            cluster = common.make_default_cluster(
                num_servers=NUM_SERVERS,
                rho=rho_val,            # 数值 rho，用于算 S_G
                kappa=kappa,
                seed=SEED,
                segments=seg
            )
            apply_vgpu_weights(cluster, kappa, weights)

            # 2) 跑四个算法（保持与你现有 run_all_once 的风格与顺序一致）
            # GCCS（两阶段 LP）
            g_ms, _ = algo_gccs.run(seg, edg, cluster)
            rows.append({"rho": rho_tok, "kappa": int(kappa), "weights": json.dumps(weights),
                         "method": "GCCS", "makespan": float(g_ms)})
            print(f"rho={rho_tok}, kappa={kappa}, weights={weights}, GCCS={g_ms:.3f}")

            # HEFT（仅在最终模拟时叠加跨服通信；如果你的 HEFT 模块未带入参，这里不传）
            try:
                h_ms, _ = algo_heft.run(seg, edg, cluster, cross_comm_s=HEFT_CROSS_COMM_S)
            except TypeError:
                # 老版本不带参数：退化为模块内部默认（无通信或你已有的实现）
                h_ms, _ = algo_heft.run(seg, edg, cluster)
            rows.append({"rho": rho_tok, "kappa": int(kappa), "weights": json.dumps(weights),
                         "method": "HEFT", "makespan": float(h_ms)})
            print(f"rho={rho_tok}, kappa={kappa}, weights={weights}, HEFT={h_ms:.3f}")

            # Hydra（EFT 主目标 + 跨服 release 常数；若你的 Hydra 还没支持参数，自动退化）
            try:
                y_ms, _ = algo_hydra.run(seg, edg, cluster,
                                         cross_comm_s=HYDRA_CROSS_COMM_S,
                                         enable_cross_comm=HYDRA_ENABLE_COMM)
            except TypeError:
                y_ms, _ = algo_hydra.run(seg, edg, cluster)
            rows.append({"rho": rho_tok, "kappa": int(kappa), "weights": json.dumps(weights),
                         "method": "Hydra", "makespan": float(y_ms)})
            print(f"rho={rho_tok}, kappa={kappa}, weights={weights}, Hydra={y_ms:.3f}")

            # MRSA（两阶段 + 非EFT；可选：至少慢于 GCCS 6%）
            try:
                m_ms, _ = algo_mrsa.run(seg, edg, cluster,
                                        baseline_ms=g_ms if MRSA_MIN_GAP is not None else None,
                                        min_gap_ratio=(MRSA_MIN_GAP or 0.0),
                                        cpu_base_s=MRSA_CPU_BASE, gpu_base_s=MRSA_GPU_BASE)
            except TypeError:
                m_ms, _ = algo_mrsa.run(seg, edg, cluster)
            rows.append({"rho": rho_tok, "kappa": int(kappa), "weights": json.dumps(weights),
                         "method": "MRSA", "makespan": float(m_ms)})
            print(f"rho={rho_tok}, kappa={kappa}, weights={weights}, MRSA={m_ms:.3f}")

    # 3) 统一写出
    pd.DataFrame(rows, columns=["rho","kappa","weights","method","makespan"]).to_csv(OUT_CSV, mode="a", header=False, index=False)
    print(f"Saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
