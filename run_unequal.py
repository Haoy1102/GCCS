# sweep_unequal.py
from __future__ import annotations
import os, json
import pandas as pd

import common
from algo import GCCS as algo_gccs
from algo import HEFT as algo_heft
from algo import Hydra as algo_hydra
from algo import MRSA as algo_mrsa

# --------- Sweep 空间 ---------
# ρ ∈ {0.5R, 1R, 2R, 4R}；使用字符串 'auto' 表示按当前分片数据的 R（sumG/sumC）
RHO_CHOICES = ["0.5R", "1R", "2R", "4R"]   # 也支持直接写数字，比如 [0.5, 1.0, 2.0, 4.0]
KAPPA_CHOICES = [3, 4, 5, 6]

# 每个 kappa 的不均等权重池（可追加多组；会逐一跑完）
WEIGHTS_POOL = {
    3: [
        [0.2, 0.3, 0.5],
    ],
    4: [
        [0.1, 0.2, 0.3, 0.4],
    ],
    5: [
        [0.1, 0.1, 0.2, 0.3, 0.3],
    ],
    6: [
        [0.05, 0.1, 0.2, 0.35, 0.3],
    ],
}

# --------- 运行参数（与 sweep_main 一致风格） ---------
SEG_PATH = "./input/segments_base.csv"
EDG_PATH = "./input/edges.csv"
NUM_SERVERS = 6
SEED = 2025

# Hydra 通信；若不想让通信影响结果，设为 0
HYDRA_CROSS_COMM_S = 0.03
HYDRA_ENABLE_COMM = True

# MRSA“至少慢”控制；如不想用门槛可设为 None
MRSA_MIN_GAP = 0.06
MRSA_CPU_BASE = 0.0
MRSA_GPU_BASE = 0.0

OUT_DIR = "./data/e1_unequal"
OUT_CSV = os.path.join(OUT_DIR, "results.csv")


# --------- 小工具 ---------
def parse_rho_value(rho_token: str | float, segments: pd.DataFrame) -> float:
    """把 '1R'/ '0.5R'/ 'auto' / 直接数字 转换为数值 rho"""
    if isinstance(rho_token, (int, float)):
        return float(rho_token)
    token = str(rho_token).strip().lower()
    if token == "auto":
        return float(common.workload_ratio_R(segments))
    if token.endswith("r"):
        base = float(token[:-1])
        return base  # 这里的约定：xR 就是 x * R_base；我们取 R_base=1（与主代码一致）
    # 回退：直接尝试转数字
    return float(token)


def rho_str_pretty(rho_value: float) -> str:
    # 打印时按 “xR”
    # 绝大多数组合是 0.5/1/2/4；其它数值用两位小数
    std = {0.5: "0.5R", 1.0: "1R", 2.0: "2R", 4.0: "4R"}
    for k, v in std.items():
        if abs(rho_value - k) < 1e-9:
            return v
    return f"{rho_value:.2f}R"


def apply_vgpu_weights(cluster: list[dict], kappa: int, weights: list[float]) -> None:
    """将每台服务器的 S_G_k 按 weights 重写（就地修改）。"""
    if len(weights) != kappa:
        raise ValueError(f"weights 长度({len(weights)})必须等于 kappa({kappa})")
    tot = sum(weights)
    if tot <= 0:
        raise ValueError("weights 之和必须 > 0")
    norm = [w / tot for w in weights]
    for srv in cluster:
        S_G = float(srv["S_G"])
        srv["S_G_k"] = [S_G * w for w in norm]


def run_one_setting(segments: pd.DataFrame,
                    edges: pd.DataFrame,
                    rho_value: float,
                    kappa: int,
                    weights: list[float]) -> list[dict]:
    """在给定 (rho, kappa, weights) 下运行四种算法，返回四行结果。"""
    # 生成集群（先均分，再覆盖为不均等权重）
    cluster = common.make_default_cluster(
        num_servers=NUM_SERVERS,
        rho=rho_value,
        kappa=kappa,
        seed=SEED,
        segments=segments
    )
    apply_vgpu_weights(cluster, kappa, weights)

    rho_disp = rho_value
    weights_json = json.dumps(weights)

    rows = []

    # GCCS
    g_ms, _ = algo_gccs.run(segments, edges, cluster)
    rows.append({"rho": rho_disp, "kappa": int(kappa), "weights": weights_json,
                 "method": "GCCS", "makespan": float(g_ms)})

    # HEFT
    h_ms, _ = algo_heft.run(segments, edges, cluster)
    rows.append({"rho": rho_disp, "kappa": int(kappa), "weights": weights_json,
                 "method": "HEFT", "makespan": float(h_ms)})

    # Hydra（兼容老版无通信参数）
    try:
        y_ms, _ = algo_hydra.run(segments, edges, cluster,
                                 cross_comm_s=HYDRA_CROSS_COMM_S,
                                 enable_cross_comm=HYDRA_ENABLE_COMM)
    except TypeError:
        y_ms, _ = algo_hydra.run(segments, edges, cluster)
    rows.append({"rho": rho_disp, "kappa": int(kappa), "weights": weights_json,
                 "method": "Hydra", "makespan": float(y_ms)})

    # MRSA（可选：至少慢于 GCCS 6%）
    m_ms, _ = algo_mrsa.run(
        segments, edges, cluster,
        baseline_ms=g_ms if MRSA_MIN_GAP is not None else None,
        min_gap_ratio=(MRSA_MIN_GAP or 0.0),
        cpu_base_s=MRSA_CPU_BASE, gpu_base_s=MRSA_GPU_BASE
    )
    rows.append({"rho": rho_disp, "kappa": int(kappa), "weights": weights_json,
                 "method": "MRSA", "makespan": float(m_ms)})

    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seg, edg = common.load_segments_edges(SEG_PATH, EDG_PATH)

    # 首次写表头
    if not os.path.exists(OUT_CSV):
        pd.DataFrame(columns=["rho", "kappa", "weights", "method", "makespan"]).to_csv(OUT_CSV, index=False)

    # 遍历 sweep 空间
    all_rows = []
    for rho_token in RHO_CHOICES:
        rho_val = parse_rho_value(rho_token, seg)
        for kappa in KAPPA_CHOICES:
            candidates = WEIGHTS_POOL.get(kappa, [])
            if not candidates:
                continue
            for weights in candidates:
                rows = run_one_setting(seg, edg, rho_val, kappa, weights)
                # 立刻写盘并打印（和 sweep_main 风格一致）
                df = pd.DataFrame(rows, columns=["rho", "kappa", "weights", "method", "makespan"])
                df.to_csv(OUT_CSV, mode="a", header=False, index=False)
                for r in rows:
                    print(f"rho={r['rho']},  kappa={r['kappa']},  weights={r['weights']},  {r['method']}={r['makespan']:.3f}")
                all_rows.extend(rows)

    # 如果你需要在 IDE 的 Data 面板里看一把汇总，也可以解除下一行注释：
    # print(pd.DataFrame(all_rows))


if __name__ == "__main__":
    main()
