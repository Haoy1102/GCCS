
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
from common import compute_global_ranks

COMM_BASE = 0.02
COMM_COEF_CPU = 0.006
COMM_COEF_GPU = 0.012
BW_GBPS = 12.5

def _cpu_dur(s, row):  return float(row['C_TFLOP']) / s['S_C']
def _gpu_dur(s, k, row): return float(row['G_TFLOP']) / s['S_G_k'][k]

def _comm_time(prev_row: pd.Series, src: str, dst: str, comm_scale: float) -> float:
    if src == dst: return 0.0
    if 'data_gb' in prev_row.index:
        return comm_scale * (COMM_BASE + float(prev_row['data_gb'])/BW_GBPS)
    if str(prev_row['type']) == 'GPU':
        return comm_scale * (COMM_BASE + COMM_COEF_GPU*float(prev_row['G_TFLOP']))
    return comm_scale * (COMM_BASE + COMM_COEF_CPU*float(prev_row['C_TFLOP']))

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict], comm_scale: float=1.6) -> Tuple[float, Dict[str,float]]:
    ranks, succ_all, pred_all = compute_global_ranks(segments, edges, cluster)
    cpu_avail = {s['name']: 0.0 for s in cluster}
    gpu_avail = {s['name']: [0.0 for _ in s['S_G_k']] for s in cluster}
    indeg = {}
    for tid in segments['task_id'].unique():
        preds = pred_all[tid]
        nodes = set(segments[segments['task_id']==tid]['seg_id'].astype(int).tolist())
        for v in nodes:
            indeg[(tid,v)] = len(preds[v]) if v in preds else 0
    ready = [(tid,v) for (tid,v),d in indeg.items() if d==0]
    assigned = {}; finish = {}
    while ready:
        ready.sort(key=lambda tv: ranks[(tv[0], tv[1])], reverse=True)
        tid, v = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        typ = str(row['type'])
        best=None
        if typ=='CPU':
            for s in cluster:
                rls=0.0
                for u in pred_all[tid].get(v, []):
                    ft_u = finish[(tid,u)]; kind_u, s_u, _ = assigned[(tid,u)]
                    prev_row = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]
                    rls = max(rls, ft_u + _comm_time(prev_row, s_u, s['name'], comm_scale))
                st = max(cpu_avail[s['name']], rls); ft = st + _cpu_dur(s, row)
                cand = (ft, st, 'cpu', s['name'], None)
                if (best is None) or (cand < best): best = cand
        else:
            for s in cluster:
                for k in range(len(s['S_G_k'])):
                    rls=0.0
                    for u in pred_all[tid].get(v, []):
                        ft_u = finish[(tid,u)]; kind_u, s_u, _ = assigned[(tid,u)]
                        prev_row = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]
                        rls = max(rls, ft_u + _comm_time(prev_row, s_u, s['name'], comm_scale))
                    st = max(gpu_avail[s['name']][k], rls); ft = st + _gpu_dur(s, k, row)
                    cand = (ft, st, 'gpu', s['name'], k)
                    if (best is None) or (cand < best): best = cand
        ft, st, kind, sname, k = best
        if kind=='cpu': cpu_avail[sname] = ft
        else: gpu_avail[sname][k] = ft
        finish[(tid,v)] = ft; assigned[(tid,v)] = (kind, sname, k)
        for w in succ_all[tid].get(v, []):
            indeg[(tid,w)] -= 1
            if indeg[(tid,w)] == 0: ready.append((tid,w))
    per = {s['name']: float(max([cpu_avail[s['name']]] + gpu_avail[s['name']])) for s in cluster}
    overall = max(per.values()) if per else 0.0
    return float(overall), per
