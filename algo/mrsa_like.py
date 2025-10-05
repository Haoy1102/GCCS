
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
from common import compute_global_ranks

def _cpu_dur(s, row):  return float(row['C_TFLOP']) / s['S_C']
def _gpu_dur(s, k, row): return float(row['G_TFLOP']) / s['S_G_k'][k]

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]) -> Tuple[float, Dict[str,float]]:
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
    finish={}; assigned={}
    def best_c(tid,v,typ):
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        best=float('inf')
        if typ=='CPU':
            for s in cluster:
                rls=max([0.0]+[finish[(tid,u)] for u in pred_all[tid].get(v, [])])
                start=max(cpu_avail[s['name']], rls); wait=start-rls
                c=wait+_cpu_dur(s,row); best=min(best,c)
        else:
            for s in cluster:
                for k in range(len(s['S_G_k'])):
                    rls=max([0.0]+[finish[(tid,u)] for u in pred_all[tid].get(v, [])])
                    start=max(gpu_avail[s['name']][k], rls); wait=start-rls
                    c=wait+_gpu_dur(s,k,row); best=min(best,c)
        return best
    while ready:
        ready.sort(key=lambda tv: -best_c(tv[0], tv[1], str(segments[(segments['task_id']==tv[0])&(segments['seg_id']==tv[1])].iloc[0]['type'])))
        tid,v=ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]; typ=str(row['type'])
        # FREE-first placement
        if typ=='CPU':
            choices=[(max(cpu_avail[s['name']], max([0.0]+[finish.get((tid,u),0.0) for u in pred_all[tid].get(v, [])])), 'cpu', s['name'], None) for s in cluster]
        else:
            choices=[]
            for s in cluster:
                for k in range(len(s['S_G_k'])):
                    rls=max([0.0]+[finish.get((tid,u),0.0) for u in pred_all[tid].get(v, [])])
                    st=max(gpu_avail[s['name']][k], rls)
                    choices.append((st,'gpu',s['name'],k))
        st, kind, sname, k = min(choices, key=lambda x: (x[0], x[2]))
        ft = st + (_cpu_dur(next(s for s in cluster if s['name']==sname), row) if kind=='cpu' else _gpu_dur(next(s for s in cluster if s['name']==sname), k, row))
        if kind=='cpu': cpu_avail[sname]=ft
        else: gpu_avail[sname][k]=ft
        finish[(tid,v)]=ft; assigned[(tid,v)]=(kind,sname,k)
        for w in succ_all[tid].get(v, []):
            indeg[(tid,w)]-=1
            if indeg[(tid,w)]==0: ready.append((tid,w))
    per = {s['name']: float(max([cpu_avail[s['name']]] + gpu_avail[s['name']])) for s in cluster}
    overall = max(per.values()) if per else 0.0
    return float(overall), per
