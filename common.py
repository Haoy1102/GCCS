
"""
common.py
Reusable utilities for all algorithms.
- load_segments_edges(): read CSVs from ./input
- make_default_cluster(): construct synthetic cluster
- compute_crit_for_task(): upward rank (seconds) per node
- schedule_on_server(): shared single-CPU + K-vGPU simulator with pluggable priority_fn
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from pathlib import Path

# --------------- data ---------------
def load_segments_edges(segments_csv: str|None=None, edges_csv: str|None=None):
    """If paths are None, default to ./input/segments.csv and ./input/edges.csv"""
    if segments_csv is None: segments_csv = "./input/segments_base.csv"
    if edges_csv is None: edges_csv = "./input/edges.csv"
    seg = pd.read_csv(segments_csv)
    edg = pd.read_csv(edges_csv)
    for df in (seg, edg):
        df.columns = [c.strip() for c in df.columns]
    need_seg = {'task_id','seg_id','type','C_TFLOP','G_TFLOP'}
    need_edg = {'task_id','u','v'}
    if not need_seg.issubset(set(seg.columns)):
        miss = need_seg - set(seg.columns); raise ValueError(f"segments.csv missing {miss}")
    if not need_edg.issubset(set(edg.columns)):
        miss = need_edg - set(edg.columns); raise ValueError(f"edges.csv missing {miss}")
    seg['type'] = seg['type'].astype(str)
    seg['seg_id'] = seg['seg_id'].astype(int)
    edg['u'] = edg['u'].astype(int); edg['v'] = edg['v'].astype(int)
    return seg, edg

# --------------- cluster ---------------
def make_default_cluster(num_servers: int = 6,
                         cpu_range=(1.2, 2.0),
                         rho: float = 4.0,
                         k_options=(2,3,4),
                         seed: int = 2025) -> List[Dict]:
    rng = np.random.RandomState(seed)
    hetero_patterns = {
        2: [[0.5,0.5],[0.3,0.7],[0.1,0.9]],
        3: [[1/3,1/3,1/3],[0.2,0.3,0.5]],
        4: [[0.25,0.25,0.25,0.25],[0.1,0.2,0.3,0.4]],
    }
    cluster = []
    for i in range(num_servers):
        S_C = float(rng.uniform(*cpu_range))
        k = int(rng.choice(k_options))
        w = hetero_patterns.get(k, [[1.0/k]*k])[rng.randint(0, len(hetero_patterns.get(k, [[1.0/k]*k])))]
        S_G = float(rho * S_C)
        S_G_k = [S_G*wi for wi in w]
        cluster.append({'name': f'srv{i+1}', 'S_C': S_C, 'S_G': S_G, 'vgpu_weights': w, 'S_G_k': S_G_k, 'rho': float(rho)})
    return cluster

# --------------- critical path ---------------
def compute_crit_for_task(task_id: str, segments: pd.DataFrame, edges: pd.DataFrame, S_C: float, S_G_sum: float):
    segs = segments[segments['task_id']==task_id].copy()
    def w(row):
        return (row['C_TFLOP']/S_C) if row['type']=='CPU' else (row['G_TFLOP']/S_G_sum)
    segs['w'] = segs.apply(w, axis=1)

    task_edges = edges[edges['task_id']==task_id]
    succ = defaultdict(list); pred = defaultdict(list); nodes=set(segs['seg_id'].tolist())
    for _,e in task_edges.iterrows():
        u=int(e['u']); v=int(e['v'])
        succ[u].append(v); pred[v].append(u)
        nodes.add(u); nodes.add(v)

    indeg = {v: len(pred[v]) for v in nodes}
    q = deque([v for v in nodes if indeg[v]==0]); topo=[]
    while q:
        x=q.popleft(); topo.append(x)
        for y in succ[x]:
            indeg[y]-=1
            if indeg[y]==0: q.append(y)

    w_map = {int(r.seg_id): float(r.w) for r in segs.itertuples()}
    crit = {v:0.0 for v in nodes}
    for v in reversed(topo):
        crit[v] = w_map.get(v,0.0) + (max((crit[u] for u in succ[v]), default=0.0))
    return crit, succ, pred

# --------------- per-server scheduler ---------------
def schedule_on_server(server_name: str, segments: pd.DataFrame, edges: pd.DataFrame,
                       cluster: List[Dict], assigned_tasks: List[str],
                       priority_fn):
    srv = next(s for s in cluster if s['name']==server_name)
    S_C = srv['S_C']; S_G_k = srv['S_G_k']; S_G_sum = sum(S_G_k)
    T_cpu = 0.0; T_k = [0.0 for _ in S_G_k]; finish = {}

    task_struct = {}
    for tid in assigned_tasks:
        crit, succ, pred = compute_crit_for_task(tid, segments, edges, S_C, S_G_sum)
        indeg = {v: len(pred[v]) for v in set(list(crit.keys()) + list(pred.keys()))}
        for v in crit: indeg.setdefault(v,0)
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
            eft=[(T_k[k]+row['G_TFLOP']/S_G_k[k],k) for k in range(len(S_G_k))]
            end,kbest=min(eft, key=lambda x:x[0]); T_k[kbest]=end
        finish[(tid,v)] = end
        for u in task_struct[tid]['succ'][v]:
            task_struct[tid]['indeg'][u]-=1
            if task_struct[tid]['indeg'][u]==0:
                typ2 = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]['type']
                ready.append((tid,u,typ2))

    task_done={}
    for tid in assigned_tasks:
        seg_ids = segments[segments['task_id']==tid]['seg_id'].tolist()
        task_done[tid]=max(finish[(tid,v)] for v in seg_ids)
    makespan = max(task_done.values()) if task_done else 0.0
    return float(makespan), task_done

# --------------- global upward rank (average speeds) ---------------
def compute_global_ranks(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]):
    """Compute rank_u using average CPU speed and average single-vGPU speed across cluster."""
    if len(cluster)==0:
        raise ValueError("cluster is empty")
    avg_cpu = float(np.mean([s['S_C'] for s in cluster]))
    all_q = []
    for s in cluster:
        all_q.extend(list(s['S_G_k']))
    avg_gpuq = float(np.mean(all_q)) if all_q else 1.0

    seg = segments.copy()
    def w(row):
        return (row['C_TFLOP']/avg_cpu) if row['type']=='CPU' else (row['G_TFLOP']/avg_gpuq)
    seg['w'] = seg.apply(w, axis=1)

    ranks = {}
    succ = {}
    pred = {}

    for tid in seg['task_id'].unique():
        segs = seg[seg['task_id']==tid]
        task_edges = edges[edges['task_id']==tid]
        s = defaultdict(list); p = defaultdict(list); nodes=set(segs['seg_id'].tolist())
        for _,e in task_edges.iterrows():
            u=int(e['u']); v=int(e['v'])
            s[u].append(v); p[v].append(u); nodes.add(u); nodes.add(v)
        indeg = {v: len(p[v]) for v in nodes}
        q=deque([v for v in nodes if indeg[v]==0]); topo=[]
        while q:
            x=q.popleft(); topo.append(x)
            for y in s[x]:
                indeg[y]-=1
                if indeg[y]==0: q.append(y)
        w_map = {int(r.seg_id): float(r.w) for r in segs.itertuples()}
        rtask={}
        for v in reversed(topo):
            rtask[v] = w_map.get(v,0.0) + (max((rtask[u] for u in s[v]), default=0.0))
        for v,val in rtask.items():
            ranks[(tid,int(v))]=float(val)
        succ[tid]=s; pred[tid]=p
    return ranks, succ, pred

# --------------- global list scheduling ---------------

def schedule_global(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict],
                    task_priority_fn,
                    selection_policy: str = "EFT"):
    """
    Event-driven global list scheduling across all servers.
    - 'task_priority_fn(tid, v, typ, ctx)' returns a scalar priority; higher first.
      ctx includes live cpu/gpu availabilities and the node's release_time.
    - selection_policy: 'EFT' or 'FREE'
    """
    ranks, succ_all, pred_all = compute_global_ranks(segments, edges, cluster)

    cpu_avail = {s['name']: 0.0 for s in cluster}
    gpu_avail = {s['name']: [0.0 for _ in s['S_G_k']] for s in cluster}

    def cpu_dur(server_name, row):
        S_C = next(s for s in cluster if s['name']==server_name)['S_C']
        return float(row['C_TFLOP'])/S_C
    def gpu_dur(server_name, k, row):
        cap = next(s for s in cluster if s['name']==server_name)['S_G_k'][k]
        return float(row['G_TFLOP'])/cap

    indeg = {}
    release_time = {}
    for tid in segments['task_id'].unique():
        succ = succ_all[tid]; pred = pred_all[tid]
        nodes = set()
        nodes.update(segments[segments['task_id']==tid]['seg_id'].astype(int).tolist())
        for a in succ: 
            nodes.add(a)
            for b in succ[a]: nodes.add(b)
        for v in nodes:
            indeg[(tid,v)] = len(pred[v])
        for v,c in list(indeg.items()):
            if v[0]==tid and c==0:
                release_time[(tid,v[1])] = 0.0

    ready = [(tid, v) for (tid,v),deg in indeg.items() if deg==0]
    finish_time = {}

    while ready:
        def priority_of(tid,v):
            row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
            typ = str(row['type'])
            return task_priority_fn(tid, v, typ, {
                'ranks':ranks,
                'segments':segments,
                'cluster':cluster,
                'cpu_avail':cpu_avail,
                'gpu_avail':gpu_avail,
                'release_time': release_time.get((tid,v), 0.0)
            })
        ready.sort(key=lambda tv: priority_of(tv[0], tv[1]), reverse=True)
        tid,v = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        typ = str(row['type'])
        rls = release_time[(tid,v)]

        candidates=[]
        if typ=='CPU':
            for s in cluster:
                st = max(cpu_avail[s['name']], rls)
                dur = cpu_dur(s['name'], row)
                ft = st + dur
                candidates.append(('cpu', s['name'], None, st, dur, ft))
        else:
            for s in cluster:
                for k in range(len(s['S_G_k'])):
                    st = max(gpu_avail[s['name']][k], rls)
                    dur = gpu_dur(s['name'], k, row)
                    ft = st + dur
                    candidates.append(('gpu', s['name'], k, st, dur, ft))

        if selection_policy.upper()=='FREE':
            chosen = min(candidates, key=lambda x: (x[3], x[1], -1 if x[2] is None else x[2]))
        else:
            chosen = min(candidates, key=lambda x: (x[5], x[3]))

        kind, sname, k, st, dur, ft = chosen
        if kind=='cpu':
            cpu_avail[sname] = ft
        else:
            gpu_avail[sname][k] = ft
        finish_time[(tid,v)] = ft

        succ = succ_all[tid]
        for u in succ.get(v, []):
            indeg[(tid,u)] -= 1
            release_time[(tid,u)] = max(release_time.get((tid,u), 0.0), ft)
            if indeg[(tid,u)] == 0:
                ready.append((tid,u))

    per_server={}
    for s in cluster:
        ms = max([cpu_avail[s['name']]] + (gpu_avail[s['name']] if gpu_avail[s['name']] else [0.0]))
        per_server[s['name']] = float(ms)
    overall = max(per_server.values()) if per_server else 0.0
    return float(overall), per_server


# --------------- cluster enrichment (robust) ---------------
def enrich_cluster(cluster_cfg: List[Dict]) -> List[Dict]:
    """
    Ensure each server has fields: S_C, rho, S_G, S_G_k, vgpu_weights.
    Inference order:
      - if S_G and S_C exist -> rho = S_G/S_C
      - elif rho and S_C exist -> S_G = rho * S_C
      - if S_G_k missing and vgpu_weights present -> S_G_k = S_G * weights
      - if vgpu_weights missing but S_G_k present -> weights = normalize(S_G_k)
      - default queues: 4 equal weights
    """
    out=[]
    for s0 in cluster_cfg:
        s=dict(s0)
        if 'S_C' not in s or s['S_C'] is None:
            raise ValueError(f"Server missing S_C: {s0}")
        S_C = float(s['S_C'])
        rho = s.get('rho', None)
        S_G  = s.get('S_G', None)
        if S_G is not None and rho is None:
            rho = float(S_G)/S_C if S_C>0 else 0.0
        if rho is not None and S_G is None:
            S_G = float(rho)*S_C
        if rho is None and S_G is None:
            if s.get('S_G_k'):
                S_G = float(np.sum(s['S_G_k'])); rho = S_G/S_C if S_C>0 else 0.0
            else:
                rho = 4.0; S_G = rho*S_C
        weights = s.get('vgpu_weights', None)
        S_G_k   = s.get('S_G_k', None)
        if S_G_k is None and weights:
            S_G_k = [float(S_G)*float(w) for w in weights]
        if weights is None and S_G_k:
            total=float(np.sum(S_G_k))
            weights=[float(x)/total if total>0 else 0.0 for x in S_G_k]
        if S_G_k is None and weights is None:
            weights=[0.25,0.25,0.25,0.25]
            S_G_k=[float(S_G)*w for w in weights]
        s.update({'S_C':float(S_C),'rho':float(rho),'S_G':float(S_G),
                  'vgpu_weights': list(map(float,weights)),
                  'S_G_k': list(map(float,S_G_k))})
        out.append(s)
    return out

# --------------- lower bound for phase-1 ---------------
def lower_bound_z(totals: pd.DataFrame, cluster: List[Dict]) -> float:
    SC_sum = sum(s['S_C'] for s in cluster)
    SG_sum = sum(s['S_G'] for s in cluster)
    lb_avg = max(totals['total_C'].sum()/SC_sum,
                 totals['total_G'].sum()/SG_sum)
    lb_jobs = 0.0
    for r in totals.itertuples():
        best = min(max(r.total_C/s['S_C'], r.total_G/s['S_G']) for s in cluster)
        lb_jobs = max(lb_jobs, best)
    return max(lb_avg, lb_jobs)
