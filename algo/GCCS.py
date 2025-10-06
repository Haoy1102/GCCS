# algo/GCCS.py
# GCCS (your two-phase algorithm)
# Phase-1: LP (PuLP) min-max assignment + capacity-aware randomized rounding (robust normalization)
# Phase-2: per-server scheduling using common.schedule_on_server
# --------- 本版新增：候选分数 Πcpu/Πgpu（含 β 参数），由 priority_fn 实现 ---------

from __future__ import annotations
from typing import List, Dict, Tuple, Callable
import numpy as np
import pandas as pd
from common import schedule_on_server

# ---------- Phase-1: LP (PuLP) ----------
def phase1_lp(
    segments: pd.DataFrame,
    cluster: List[Dict],
    seed: int = 2025,
    slack: float = 1.08,
) -> Tuple[Dict[str, Dict], float]:
    try:
        import pulp
    except Exception as e:
        raise RuntimeError("GCCS Phase-1 需要 PuLP，请先安装: pip install pulp") from e

    totals = (
        segments.groupby('task_id')
        .agg(total_C=('C_TFLOP','sum'), total_G=('G_TFLOP','sum'))
        .reset_index()
    )
    tasks = totals['task_id'].tolist()
    N = len(cluster)
    Ti = totals.set_index('task_id')

    prob = pulp.LpProblem('gccs_phase1', pulp.LpMinimize)
    x = {(i,n): pulp.LpVariable(f"x_{i}_{n}", lowBound=0, upBound=1) for i in tasks for n in range(N)}
    z = pulp.LpVariable("z", lowBound=0)
    prob += z

    for i in tasks:
        prob += pulp.lpSum(x[(i,n)] for n in range(N)) == 1
    for n, srv in enumerate(cluster):
        prob += pulp.lpSum((Ti.loc[i,'total_C']/srv['S_C']) * x[(i,n)] for i in tasks) <= z
        prob += pulp.lpSum((Ti.loc[i,'total_G']/srv['S_G']) * x[(i,n)] for i in tasks) <= z

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    z_star = pulp.value(z)
    if z_star is None:
        raise RuntimeError("PuLP 求解失败: z=None")
    z_cap = float(z_star) * float(slack)

    # 容量感知随机舍入（健壮的概率归一化）
    xhat = {
        i: np.clip(np.array([float(pulp.value(x[(i,n)]) or 0.0) for n in range(N)], dtype=float), 0.0, 1.0)
        for i in tasks
    }
    order = sorted(tasks, key=lambda i: max(Ti.loc[i,'total_C'], Ti.loc[i,'total_G']), reverse=True)

    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    rng = np.random.RandomState(seed)

    for i in order:
        base_p = xhat[i].copy()
        ssum = float(base_p.sum())
        prob_vec = (base_p/ssum) if ssum>1e-12 else (np.ones(N)/N)

        assigned=None
        mask = np.ones(N, dtype=bool)
        for _ in range(N):
            p = prob_vec * mask
            ps = float(p.sum())
            if ps <= 1e-12:
                break
            p /= ps

            n = int(rng.choice(N, p=p))
            srv = cluster[n]; name = srv['name']
            c_after = state[name]['cpu_load'] + Ti.loc[i,'total_C']/srv['S_C']
            g_after = state[name]['gpu_load'] + Ti.loc[i,'total_G']/srv['S_G']
            if (c_after <= z_cap) and (g_after <= z_cap):
                state[name]['cpu_load'] = c_after
                state[name]['gpu_load'] = g_after
                state[name]['tasks'].append(i)
                assigned = name
                break
            mask[n] = False

        if assigned is None:
            # 最小违约修复
            best_name, best_metric = None, float('inf')
            for srv in cluster:
                nm = srv['name']
                c_after = state[nm]['cpu_load'] + Ti.loc[i,'total_C']/srv['S_C']
                g_after = state[nm]['gpu_load'] + Ti.loc[i,'total_G']/srv['S_G']
                metric = max(c_after, g_after)
                if metric < best_metric:
                    best_metric, best_name = metric, nm
            best_srv = next(s for s in cluster if s['name']==best_name)
            state[best_name]['cpu_load'] += Ti.loc[i,'total_C']/best_srv['S_C']
            state[best_name]['gpu_load'] += Ti.loc[i,'total_G']/best_srv['S_G']
            state[best_name]['tasks'].append(i)

    return state, z_cap

def phase1_random(segments, cluster, seed=2025):
    """去掉 min-max 打包：完全随机把每个任务指派到一台服务器（不看任何负载）。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    totals = (segments.groupby('task_id')
              .agg(total_C=('C_TFLOP','sum'), total_G=('G_TFLOP','sum'))
              .reset_index())
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    for r in totals.itertuples():
        srv = cluster[int(rng.randint(0, len(cluster)))]
        state[srv['name']]['tasks'].append(r.task_id)
    return state, 0.0

def phase1_global_greedy(segments, cluster):
    """去掉 min-max 打包：逐任务独立选 argmin_s max(C/S_C[s], G/S_G[s])，忽略当前负载。"""
    totals = (segments.groupby('task_id')
              .agg(total_C=('C_TFLOP','sum'), total_G=('G_TFLOP','sum'))
              .reset_index())
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    for r in totals.itertuples():
        best = None; best_cost = float('inf')
        for srv in cluster:
            cost = max(r.total_C/srv['S_C'], r.total_G/srv['S_G'])
            if cost < best_cost:
                best_cost = cost; best = srv['name']
        state[best]['tasks'].append(r.task_id)
    return state, 0.0

# ---------- Phase-2: priority(含 β) + per-server 调度 ----------
def make_priority_fn(beta_cpu: float = 0.2,
                     beta_gpu: float = 0.2,
                     use_time: bool = True) -> Callable:
    """
    返回 priority_fn(tid, v, ctx)：
      Π_cpu(i,s) = Crit(i,s) - β * C_{i,s}
      Π_gpu(i,s) = Crit(i,s) + β' * G_{i,s}
    其中 C_{i,s}, G_{i,s} 默认用 “时间惩罚”（C/S_C 与 G/avg(S_Gk)），更稳定；
    若 use_time=False 则用原始 TFLOP（C 与 G）。
    """
    def prio(tid, v, ctx):
        srv  = ctx['server']
        segs = ctx['segments']
        crit = ctx['task_struct'][tid]['crit'][v]
        row  = segs[(segs['task_id']==tid) & (segs['seg_id']==v)].iloc[0]
        typ  = str(row['type']).upper()

        if use_time:
            if typ == 'CPU':
                C_is = float(row['C_TFLOP']) / float(srv['S_C'])
                return crit - beta_cpu * C_is
            else:
                G_is = float(row['G_TFLOP']) / float(np.mean(srv['S_G_k']))
                return crit + beta_gpu * G_is
        else:
            if typ == 'CPU':
                return crit - beta_cpu * float(row['C_TFLOP'])
            else:
                return crit + beta_gpu * float(row['G_TFLOP'])
    return prio


def run(
    segments: pd.DataFrame,
    edges: pd.DataFrame,
    cluster: List[Dict],
    seed: int = 2025,
    slack: float = 1.08,
    beta_cpu: float = 1.2,
    beta_gpu: float = 0.5,
    phase1_mode: str = "lp",  # 新增：'lp' | 'random' | 'global'
    gpu_queue_mode: str = "dynamic",  # 新增：'dynamic' | 'fixed'
    prio_use_time: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Phase-1：LP 指派；Phase-2：在每台服务器上用 Π 分数排序（含 β），
    CPU 串行、GPU 按 EFT 选 vGPU 队列（schedule_on_server 已实现）。
    """
    # server_state, _ = phase1_lp(segments, cluster, seed=seed, slack=slack)
    if phase1_mode == "lp":
        server_state, z = phase1_lp(segments, cluster, seed=seed, slack=slack)
    elif phase1_mode == "random":
        server_state, z = phase1_random(segments, cluster, seed=seed)
    elif phase1_mode == "global":
        server_state, z = phase1_global_greedy(segments, cluster)
    else:
        raise ValueError(f"Unknown phase1_mode = {phase1_mode}")


    per = {}
    for s in cluster:
        prio = make_priority_fn(beta_cpu=beta_cpu, beta_gpu=beta_gpu, use_time=prio_use_time)
        ms, _ = schedule_on_server(
            s['name'], segments, edges, cluster,
            server_state[s['name']]['tasks'],
            priority_fn=prio,
            gpu_queue_mode = gpu_queue_mode
        )
        per[s['name']] = ms

    overall = max(per.values()) if per else 0.0
    return float(overall), per
