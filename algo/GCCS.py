# algo/GCCS.py
# GCCS (your two-phase algorithm)
# Phase-1: LP (PuLP) min-max assignment + capacity-aware randomized rounding (robust normalization)
# Phase-2: per-server scheduling using common.schedule_on_server

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from common import schedule_on_server

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

def run(
    segments: pd.DataFrame,
    edges: pd.DataFrame,
    cluster: List[Dict],
    seed: int = 2025,
    slack: float = 1.08
) -> Tuple[float, Dict[str, float]]:
    server_state, z = phase1_lp(segments, cluster, seed=seed, slack=slack)

    def prio(tid, v, ctx):
        # use critical-path based priority computed in schedule_on_server
        return ctx['task_struct'][tid]['crit'][v]

    per = {}
    for s in cluster:
        ms, _ = schedule_on_server(s['name'], segments, edges, cluster, server_state[s['name']]['tasks'], prio)
        per[s['name']] = ms
    overall = max(per.values()) if per else 0.0
    return float(overall), per
