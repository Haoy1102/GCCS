
"""
algo/two_phase.py
GCCS-2Phase (user's two-stage algorithm), refactored to reuse common utilities:
- cluster enrichment: common.enrich_cluster
- lower bound: common.lower_bound_z
- per-server scheduling (CPU-serial + GPU-EFT with Crit priority): common.schedule_on_server
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from common import enrich_cluster, lower_bound_z, schedule_on_server

# ---------- Phase-1: greedy ----------
def phase1_greedy(segments: pd.DataFrame, cluster: List[Dict]):
    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    lookup = {s['name']: s for s in cluster}
    order = totals.sort_values(['total_G','total_C'], ascending=[False, False])

    for r in order.itertuples():
        best_srv, best_metric = None, float('inf')
        for srv in cluster:
            cpu_t = state[srv['name']]['cpu_load'] + r.total_C/srv['S_C']
            gpu_t = state[srv['name']]['gpu_load'] + r.total_G/srv['S_G']
            metric = max(cpu_t, gpu_t)
            if metric < best_metric:
                best_metric, best_srv = metric, srv['name']
        state[best_srv]['cpu_load'] += r.total_C / lookup[best_srv]['S_C']
        state[best_srv]['gpu_load'] += r.total_G / lookup[best_srv]['S_G']
        state[best_srv]['tasks'].append(r.task_id)

    z_est = max(max(v['cpu_load'], v['gpu_load']) for v in state.values())
    return state, z_est

# ---------- Phase-1: relax (solver-free) ----------
def phase1_relax(segments: pd.DataFrame, cluster: List[Dict], seed=2025, slack=1.08):
    rng = np.random.RandomState(seed)
    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    z = lower_bound_z(totals, cluster) * slack

    prefs = {}
    for r in totals.itertuples():
        costs = np.array([max(r.total_C/s['S_C'], r.total_G/s['S_G']) for s in cluster], dtype=float)
        prob  = (1.0/(costs+1e-12)); prob = prob/prob.sum()
        prefs[r.task_id] = prob

    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}

    S_C_avg=np.mean([s['S_C'] for s in cluster]); S_G_avg=np.mean([s['S_G'] for s in cluster])
    order = totals.copy()
    order['key'] = np.maximum(order['total_C']/S_C_avg, order['total_G']/S_G_avg)
    order = order.sort_values('key', ascending=False)

    for r in order.itertuples():
        prob = prefs[r.task_id].copy()
        assigned=None
        for _ in range(len(cluster)):
            mask = np.ones(len(cluster), dtype=bool)
            for idx,srv in enumerate(cluster):
                c_after = state[srv['name']]['cpu_load'] + r.total_C/srv['S_C']
                g_after = state[srv['name']]['gpu_load'] + r.total_G/srv['S_G']
                if (c_after>z) or (g_after>z): mask[idx]=False
            if not mask.any(): break
            p = prob*mask; p = p/p.sum()
            choice = rng.choice(len(cluster), p=p)
            srv = cluster[choice]
            c_after = state[srv['name']]['cpu_load'] + r.total_C/srv['S_C']
            g_after = state[srv['name']]['gpu_load'] + r.total_G/srv['S_G']
            if (c_after<=z) and (g_after<=z):
                state[srv['name']]['cpu_load']=c_after
                state[srv['name']]['gpu_load']=g_after
                state[srv['name']]['tasks'].append(r.task_id)
                assigned = srv['name']; break
        if assigned is None:
            best_srv, best_metric = None, float('inf')
            for srv in cluster:
                c_after = state[srv['name']]['cpu_load'] + r.total_C/srv['S_C']
                g_after = state[srv['name']]['gpu_load'] + r.total_G/srv['S_G']
                metric = max(c_after, g_after)
                if metric < best_metric: best_metric, best_srv = metric, srv['name']
            state[best_srv]['cpu_load'] += r.total_C / next(s['S_C'] for s in cluster if s['name']==best_srv)
            state[best_srv]['gpu_load'] += r.total_G / next(s['S_G'] for s in cluster if s['name']==best_srv)
            state[best_srv]['tasks'].append(r.task_id)
    return state, z

# ---------- Phase-1: lp (PuLP) ----------
def phase1_lp(segments: pd.DataFrame, cluster: List[Dict], seed=2025, slack=1.05):
    try:
        import pulp
    except Exception:
        return phase1_relax(segments, cluster, seed=seed, slack=slack)

    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    tasks = totals['task_id'].tolist()
    N = len(cluster)

    prob = pulp.LpProblem('phase1_lp', pulp.LpMinimize)
    x = {(i,n): pulp.LpVariable(f"x_{i}_{n}", lowBound=0, upBound=1) for i in tasks for n in range(N)}
    z = pulp.LpVariable("z", lowBound=0)
    prob += z

    Ti = totals.set_index('task_id')
    for i in tasks:
        prob += pulp.lpSum(x[(i,n)] for n in range(N)) == 1
    for n,srv in enumerate(cluster):
        prob += pulp.lpSum((Ti.loc[i,'total_C']/srv['S_C'])*x[(i,n)] for i in tasks) <= z
        prob += pulp.lpSum((Ti.loc[i,'total_G']/srv['S_G'])*x[(i,n)] for i in tasks) <= z

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    xhat = {i: np.array([pulp.value(x[(i,n)]) for n in range(N)], dtype=float) for i in tasks}
    zstar = float(pulp.value(z)) if pulp.value(z) is not None else lower_bound_z(totals, cluster)
    z_cap = zstar * slack

    rng = np.random.RandomState(seed)
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    S_C_avg=np.mean([s['S_C'] for s in cluster]); S_G_avg=np.mean([s['S_G'] for s in cluster])
    hardness = [(i, max(Ti.loc[i,'total_C']/S_C_avg, Ti.loc[i,'total_G']/S_G_avg)) for i in tasks]
    hardness.sort(key=lambda t: t[1], reverse=True)

    for i,_ in hardness:
        prob_vec = xhat[i].copy()
        prob_vec = prob_vec/prob_vec.sum() if prob_vec.sum()>0 else np.ones(N)/N
        assigned=None
        C_i, G_i = Ti.loc[i,'total_C'], Ti.loc[i,'total_G']
        for _ in range(N):
            mask=np.ones(N,dtype=bool)
            for n,srv in enumerate(cluster):
                c_after = state[srv['name']]['cpu_load'] + C_i/srv['S_C']
                g_after = state[srv['name']]['gpu_load'] + G_i/srv['S_G']
                if (c_after>z_cap) or (g_after>z_cap): mask[n]=False
            if not mask.any(): break
            p = prob_vec*mask; p = p/p.sum()
            n_choice = rng.choice(N, p=p); srv = cluster[n_choice]
            c_after = state[srv['name']]['cpu_load'] + C_i/srv['S_C']
            g_after = state[srv['name']]['gpu_load'] + G_i/srv['S_G']
            if (c_after<=z_cap) and (g_after<=z_cap):
                state[srv['name']]['cpu_load']=c_after
                state[srv['name']]['gpu_load']=g_after
                state[srv['name']]['tasks'].append(i)
                assigned=srv['name']; break
        if assigned is None:
            best_srv, best_metric = None, float('inf')
            for srv in cluster:
                c_after = state[srv['name']]['cpu_load'] + C_i/srv['S_C']
                g_after = state[srv['name']]['gpu_load'] + G_i/srv['S_G']
                metric = max(c_after, g_after)
                if metric < best_metric: best_metric, best_srv = metric, srv['name']
            state[best_srv]['cpu_load'] += C_i / next(s['S_C'] for s in cluster if s['name']==best_srv)
            state[best_srv]['gpu_load'] += G_i / next(s['S_G'] for s in cluster if s['name']==best_srv)
            state[best_srv]['tasks'].append(i)
    return state, z_cap

# ---------- Driver ----------
def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster_cfg: List[Dict],
        seed: int = 2025, phase1: str = 'lp', slack: float = 1.08):
    cluster = enrich_cluster(cluster_cfg)

    # Phase-1: assignment
    if phase1 == 'greedy':
        server_state, z = phase1_greedy(segments, cluster)
    elif phase1 == 'relax':
        server_state, z = phase1_relax(segments, cluster, seed=seed, slack=slack)
    else:
        server_state, z = phase1_lp(segments, cluster, seed=seed, slack=slack)

    # Phase-2: per-server scheduling with Crit + EFT (reuse common.schedule_on_server)
    def prio(tid, v, ctx):
        st = ctx['task_struct'][tid]
        segs = ctx['segments']
        row = segs[(segs['task_id']==tid)&(segs['seg_id']==v)].iloc[0]
        c = st['crit'][v]
        # 留好接口：如需 beta/beta' 惩罚可在这里改
        return c

    per_server = {}
    for s in cluster:
        assigned = server_state[s['name']]['tasks']
        ms, _ = schedule_on_server(s['name'], segments, edges, cluster, assigned, prio)
        per_server[s['name']] = ms

    overall = max(per_server.values()) if per_server else 0.0
    return float(overall), per_server
