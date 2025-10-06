
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- IO ----------
def load_segments_edges(segments_csv: str|None=None, edges_csv: str|None=None):
    if segments_csv is None: segments_csv = "./input/segments_base.csv"
    if edges_csv is None: edges_csv = "./input/edges.csv"
    seg = pd.read_csv(segments_csv)
    edg = pd.read_csv(edges_csv)
    for df in (seg, edg):
        df.columns = [c.strip() for c in df.columns]
    return seg, edg

# ---------- workload ratio ----------
def workload_ratio_R(segments: pd.DataFrame) -> float:
    total_C = float(segments['C_TFLOP'].sum())
    total_G = float(segments['G_TFLOP'].sum())
    if total_C <= 1e-9 and total_G <= 1e-9:
        return 1.0
    if total_C <= 1e-9:
        return 4.0
    return total_G / total_C

# ---------- cluster builders ----------
def make_default_cluster(num_servers: int = 6,
                         cpu_range=(1.2, 2.0),
                         rho: float | str = 4.0,
                         kappa: int | None = None,
                         seed: int = 2025,
                         segments: pd.DataFrame | None = None) -> List[Dict]:
    rng = np.random.RandomState(seed)
    # use equal weights if kappa is given, else random k in {2,3,4}
    def equal_weights(k): return [1.0/k]*k
    if rho == 'auto':
        assert segments is not None, "make_default_cluster(rho='auto') requires segments"
        R = workload_ratio_R(segments)
        rho_pool = [0.5*R, 1.0*R, 2.0*R, 4.0*R]
    else:
        rho_pool = [float(rho)]
    cluster = []
    for i in range(num_servers):
        S_C = float(rng.uniform(*cpu_range))
        if kappa is None:
            k = int(rng.choice([2,3,4]))
        else:
            k = int(kappa)
        w = equal_weights(k)
        rho_i = rho_pool[i % len(rho_pool)]
        S_G = float(rho_i * S_C)
        S_G_k = [S_G*wi for wi in w]
        cluster.append({'name': f'srv{i+1}', 'S_C': S_C, 'S_G': S_G, 'vgpu_weights': w, 'S_G_k': S_G_k, 'rho': float(rho_i)})
    return cluster

def adjust_cluster_rho_by_workload(segments: pd.DataFrame, cluster: List[Dict]) -> List[Dict]:
    R = workload_ratio_R(segments)
    rho_pool = [0.5*R, 1.0*R, 2.0*R, 4.0*R]
    out=[]
    for i,s0 in enumerate(cluster):
        s=dict(s0)
        s['rho'] = float(rho_pool[i % len(rho_pool)])
        s['S_G'] = float(s['rho'] * s['S_C'])
        if 'vgpu_weights' in s and s['vgpu_weights']:
            s['S_G_k'] = [s['S_G'] * float(w) for w in s['vgpu_weights']]
        elif 'S_G_k' in s and s['S_G_k']:
            tot = float(sum(s['S_G_k'])) or 1.0
            w = [x/tot for x in s['S_G_k']]
            s['vgpu_weights'] = w
            s['S_G_k'] = [s['S_G'] * wi for wi in w]
        else:
            s['vgpu_weights'] = [0.25,0.25,0.25,0.25]
            s['S_G_k'] = [s['S_G']*w for w in s['vgpu_weights']]
        out.append(s)
    return out

# ---------- critical-path per task ----------
def compute_crit_for_task(task_id: str, segments: pd.DataFrame, edges: pd.DataFrame, S_C: float, S_G_sum: float):
    segs = segments[segments['task_id']==task_id].copy()
    segs['w'] = segs.apply(lambda r: (r['C_TFLOP']/S_C) if r['type']=='CPU' else (r['G_TFLOP']/S_G_sum), axis=1)
    task_edges = edges[edges['task_id']==task_id]
    succ = defaultdict(list); pred = defaultdict(list); nodes=set(segs['seg_id'].astype(int).tolist())
    for _,e in task_edges.iterrows():
        u=int(e['u']); v=int(e['v'])
        succ[u].append(v); pred[v].append(u); nodes.add(u); nodes.add(v)
    indeg = {v: len(pred[v]) for v in nodes}
    q=deque([v for v in nodes if indeg[v]==0]); topo=[]
    while q:
        x=q.popleft(); topo.append(x)
        for y in succ[x]:
            indeg[y]-=1
            if indeg[y]==0: q.append(y)
    crit = {v:0.0 for v in nodes}
    w_map = {int(r.seg_id): float(r.w) for r in segs.itertuples()}
    for v in reversed(topo):
        crit[v] = w_map.get(v,0.0) + (max((crit[u] for u in succ[v]), default=0.0))
    return crit, succ, pred

# ---------- per-server scheduler (CPU-serial + GPU-EFT) ----------
def schedule_on_server(server_name: str, segments: pd.DataFrame, edges: pd.DataFrame,
                       cluster: List[Dict], assigned_tasks: List[str],
                       priority_fn: Callable):
    srv = next(s for s in cluster if s['name']==server_name)
    S_C = srv['S_C']; S_G_k = srv['S_G_k']; S_G_sum = sum(S_G_k)
    T_cpu = 0.0; T_k = [0.0 for _ in S_G_k]; finish = {}
    task_struct = {}
    for tid in assigned_tasks:
        crit, succ, pred = compute_crit_for_task(tid, segments, edges, S_C, S_G_sum)
        indeg = {v: len(pred[v]) for v in set(list(crit.keys()) + list(pred.keys()))}
        for v in crit: indeg.setdefault(v,0); succ.setdefault(v,[]); pred.setdefault(v,[])
        task_struct[tid] = {'crit':crit,'succ':succ,'pred':pred,'indeg':indeg}
    ready = []
    for tid in assigned_tasks:
        for v,cnt in task_struct[tid]['indeg'].items():
            if cnt==0:
                typ = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]['type']
                ready.append((tid,v,typ))
    while ready:
        ready.sort(key=lambda x: priority_fn(x[0], x[1], {'server':srv,'task_struct':task_struct,'segments':segments}), reverse=True)
        tid,v,typ = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        if typ=='CPU':
            dur = row['C_TFLOP']/S_C; start=T_cpu; end=start+dur; T_cpu=end
        else:
            preds = task_struct[tid]['pred'][v]
            rls = max([0.0]+[finish[(tid,u)] for u in preds]) if preds else 0.0
            best_k, best_end = None, float('inf')
            for k,cap in enumerate(S_G_k):
                st = max(T_k[k], rls); ft = st + row['G_TFLOP']/cap
                if ft < best_end: best_end, best_k = ft, k
            T_k[best_k] = best_end; end = best_end
        finish[(tid,v)] = end
        for u in task_struct[tid]['succ'][v]:
            task_struct[tid]['indeg'][u]-=1
            if task_struct[tid]['indeg'][u]==0:
                typ2 = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]['type']
                ready.append((tid,u,typ2))
    task_done={}
    for tid in assigned_tasks:
        seg_ids = segments[segments['task_id']==tid]['seg_id'].tolist()
        if seg_ids: task_done[tid]=max(finish[(tid,v)] for v in seg_ids)
    makespan = max(task_done.values()) if task_done else 0.0
    return float(makespan), task_done

# ---------- global ranks & list scheduling (HEFT/Hydra/MRSA) ----------
def compute_global_ranks(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]):
    avg_cpu = float(np.mean([s['S_C'] for s in cluster]))
    all_q = []
    for s in cluster: all_q.extend(list(s['S_G_k']))
    avg_gpuq = float(np.mean(all_q)) if all_q else 1.0
    seg = segments.copy()
    seg['w'] = seg.apply(lambda r: (r['C_TFLOP']/avg_cpu) if r['type']=='CPU' else (r['G_TFLOP']/avg_gpuq), axis=1)
    ranks = {}; succ_all={}; pred_all={}
    for tid in seg['task_id'].unique():
        segs = seg[seg['task_id']==tid]
        e = edges[edges['task_id']==tid]
        succ = defaultdict(list); pred = defaultdict(list); nodes=set(segs['seg_id'].astype(int).tolist())
        for _,row in e.iterrows():
            u=int(row['u']); v=int(row['v']); succ[u].append(v); pred[v].append(u); nodes.add(u); nodes.add(v)
        indeg = {v: len(pred[v]) for v in nodes}
        q=deque([v for v in nodes if indeg[v]==0]); topo=[]
        while q:
            x=q.popleft(); topo.append(x)
            for y in succ[x]:
                indeg[y]-=1
                if indeg[y]==0: q.append(y)
        w_map = {int(r.seg_id): float(r.w) for r in segs.itertuples()}
        rtask={}
        for v in reversed(topo):
            rtask[v] = w_map.get(v,0.0) + (max((rtask[u] for u in succ[v]), default=0.0))
        for v,val in rtask.items(): ranks[(tid,int(v))]=float(val)
        succ_all[tid]=succ; pred_all[tid]=pred
    return ranks, succ_all, pred_all
