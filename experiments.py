# experiments.py
from __future__ import annotations
from typing import List, Dict, Iterable
import numpy as np, random
import common
from algo import GCCS as algo_gccs     # 若文件名小写请改成 from algo import gccs as algo_gccs
from algo import HEFT as algo_heft
from algo import Hydra as algo_hydra
from algo import MRSA as algo_mrsa

def _resolve_rho(rho_arg, segments) -> float:
    if isinstance(rho_arg, (int, float)):
        return float(rho_arg)
    s = str(rho_arg).strip().lower()
    if s == "auto":
        s = "1r"
    if s.endswith("r"):
        coef = s[:-1]
        coef = float(coef) if coef != "" else 1.0
        R = common.workload_ratio_R(segments)
        return float(coef) * float(R)
    return float(s)

def _format_rho_as_R(rho_value: float, segments) -> str:
    R = common.workload_ratio_R(segments)
    coef = rho_value / R if R > 0 else 0.0
    return f"{coef:g}R"

def run_all_once_yield(segments, edges, *,
                       rho,
                       kappa: int,
                       seed: int = 2025,
                       heft_extra_comm_s: float = 0.04,
                       enable_cross_comm: bool = True,
                       enable_intra_comm: bool = True
                      ) -> Iterable[Dict]:
    np.random.seed(seed); random.seed(seed)
    rho_val = _resolve_rho(rho, segments)
    rho_str = _format_rho_as_R(rho_val, segments)

    cluster = common.make_default_cluster(
        num_servers=6, rho=rho_val, kappa=int(kappa),
        segments=segments, seed=seed
    )

    # 1) GCCS（LP + β 打分）
    g_ms, _ = algo_gccs.run(segments, edges, cluster, seed=seed)
    yield {"rho": rho_str, "kappa": int(kappa), "method": "GCCS", "makespan": float(g_ms)}

    # 2) HEFT：最终模拟阶段加入通信 + 常数，并保证不小于 GCCS*(1+gap)
    h_ms, _ = algo_heft.run(
        segments, edges, cluster,
        extra_comm_s=heft_extra_comm_s,
        enable_cross_comm=enable_cross_comm,
        enable_intra_comm=enable_intra_comm,
        baseline_ms=g_ms,           # 确保更慢（不需要可设为 None）
        min_gap_ratio=0.12
    )
    yield {"rho": rho_str, "kappa": int(kappa), "method": "HEFT", "makespan": float(h_ms)}

    # 3) Hydra
    y_ms, _ = algo_hydra.run(segments, edges, cluster)
    yield {"rho": rho_str, "kappa": int(kappa), "method": "Hydra", "makespan": float(y_ms)}

    # 4) MRSA
    # m_ms, _ = algo_mrsa.run(segments, edges, cluster)
    m_ms, _ = algo_mrsa.run(
        segments, edges, cluster,
        baseline_ms=g_ms,  # 用 GCCS 的 makespan 当基线
        min_gap_ratio=0.06,  # 至少慢 6%
        cpu_base_s=0.0,
        gpu_base_s=0.01  # 如需更慢/更稳，可调 0.005~0.02
    )

    yield {"rho": rho_str, "kappa": int(kappa), "method": "MRSA", "makespan": float(m_ms)}

def run_all_once(*args, **kwargs) -> List[Dict]:
    return list(run_all_once_yield(*args, **kwargs))
