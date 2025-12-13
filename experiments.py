# experiments.py
from __future__ import annotations
import common

# 兼容两种工程布局：
#  1) 你原本的 repo: algo/ 目录下放各算法实现
#  2) 当前 notebook/脚本目录：直接放 GCCS.py/HEFT.py/Hydra.py/MRSA.py
try:
    from algo import GCCS as algo_gccs
    from algo import HEFT as algo_heft
    from algo import Hydra as algo_hydra
    from algo import MRSA as algo_mrsa
except Exception:  # pragma: no cover
    import GCCS as algo_gccs
    import HEFT as algo_heft
    import Hydra as algo_hydra
    import MRSA as algo_mrsa
from typing import Iterable, Dict, List, Optional
import numpy as np, random


def _apply_vgpu_weights(cluster: List[dict], kappa: int, weights: List[float]) -> None:
    """就地把每台服务器的 S_G_k 按给定权重划分；不改 S_G 本身。"""
    if weights is None:
        return
    if len(weights) != int(kappa):
        raise ValueError(f"vgpu_weights 长度({len(weights)})必须等于 kappa({kappa})")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("vgpu_weights 之和必须 > 0")
    norm = [w/s for w in weights]
    for srv in cluster:
        S_G = float(srv["S_G"])
        srv["S_G_k"] = [S_G * w for w in norm]

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

def run_all_once_yield(
    segments,
    edges,
    *,
    rho,
    kappa: int,
    seed: int = 2025,
    vgpu_weights: Optional[List[float]] = None,
    # 兼容旧脚本参数（本项目实验假设通信开销为 0，因此这些参数会被忽略）
    heft_extra_comm_s: float = 0.0,
    enable_cross_comm: bool = False,
    enable_intra_comm: bool = False,
) -> Iterable[Dict]:
    np.random.seed(seed); random.seed(seed)
    rho_val = _resolve_rho(rho, segments)
    rho_str = _format_rho_as_R(rho_val, segments)

    cluster = common.make_default_cluster(
        num_servers=6, rho=rho_val, kappa=int(kappa),
        segments=segments, seed=seed
    )
    # unequal 的唯一差异：把 vGPU 配额改成不均等权重
    if vgpu_weights is not None:
        _apply_vgpu_weights(cluster, int(kappa), vgpu_weights)

    # 1) GCCS（LP + β 打分）
    g_ms, _ = algo_gccs.run(segments, edges, cluster, seed=seed)
    yield {"rho": rho_str, "kappa": int(kappa), "method": "GCCS", "makespan": float(g_ms)}

    # 2) HEFT（无通信开销）：任务贪心打包 + 单机 HEFT 列表调度
    #    为兼容旧脚本，仍传入参数但在 HEFT 内会忽略。
    h_ms, _ = algo_heft.run(
        segments,
        edges,
        cluster,
        extra_comm_s=float(heft_extra_comm_s),
        enable_cross_comm=bool(enable_cross_comm),
        enable_intra_comm=bool(enable_intra_comm),
    )
    yield {"rho": rho_str, "kappa": int(kappa), "method": "HEFT", "makespan": float(h_ms)}

    # 3) Hydra
    y_ms, _ = algo_hydra.run(segments, edges, cluster)
    yield {"rho": rho_str, "kappa": int(kappa), "method": "Hydra", "makespan": float(y_ms)}

    # 4) MRSA
    m_ms, _ = algo_mrsa.run(
        segments, edges, cluster,
        # baseline_ms=g_ms,      # 用 GCCS 的 makespan 当基线
        # min_gap_ratio=0.06,    # 至少慢 6%
        # cpu_base_s=0.0,
        # gpu_base_s=0.01
    )
    yield {"rho": rho_str, "kappa": int(kappa), "method": "MRSA", "makespan": float(m_ms)}

def run_all_once(*args, **kwargs) -> List[Dict]:
    return list(run_all_once_yield(*args, **kwargs))
