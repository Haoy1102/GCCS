
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from common import schedule_on_server, workload_ratio_R, make_default_cluster

# ---- Phase-1: minimal (greedy) + (relax/LP 可继续加) ----
def phase1_greedy(segments: pd.DataFrame, cluster: List[Dict]):
    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    best_of = {s['name']: s for s in cluster}
    order = totals.sort_values(['total_G','total_C'], ascending=[False, False])
    for r in order.itertuples():
        best_srv, best_metric = None, float('inf')
        for s in cluster:
            cpu_t = state[s['name']]['cpu_load'] + r.total_C/s['S_C']
            gpu_t = state[s['name']]['gpu_load'] + r.total_G/s['S_G']
            metric = max(cpu_t, gpu_t)
            if metric < best_metric: best_metric, best_srv = metric, s['name']
        state[best_srv]['cpu_load'] += r.total_C / best_of[best_srv]['S_C']
        state[best_srv]['gpu_load'] += r.total_G / best_of[best_srv]['S_G']
        state[best_srv]['tasks'].append(r.task_id)
    z_est = max(max(v['cpu_load'], v['gpu_load']) for v in state.values())
    return state, z_est

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict], phase1: str='greedy', seed: int=2025, slack: float=1.08):
    # cluster 已在 main 中按 rho='auto' 或固定 rho 构造
    if phase1=='greedy':
        server_state, z = phase1_greedy(segments, cluster)
    else:
        server_state, z = phase1_greedy(segments, cluster)
    def prio(tid, v, ctx):  # Crit 优先级（已在 schedule_on_server 内部预计算）
        return ctx['task_struct'][tid]['crit'][v]
    per = {}
    for s in cluster:
        ms,_ = schedule_on_server(s['name'], segments, edges, cluster, server_state[s['name']]['tasks'], prio)
        per[s['name']] = ms
    overall = max(per.values()) if per else 0.0
    return float(overall), per
