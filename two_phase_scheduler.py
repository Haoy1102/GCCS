"""
Two-phase CPU–GPU scheduler (user's algorithm)
Phase-1 options:
  - 'greedy' : min–max greedy packing
  - 'relax'  : solver-free LP-like soft preference + randomized rounding
  - 'lp'     : true LP relaxation solved via PuLP, then randomized rounding (fallback to 'relax' if PuLP not available)

Inputs:
  segments CSV: columns [task_id, seg_id, type{CPU,GPU}, C_TFLOP, G_TFLOP, template_id, scale, is_heavy]
  edges    CSV: columns [task_id, u, v]

Outputs:
  batch makespan (seconds) + per-server makespan & task completions
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# --------------- Paths & Config ---------------
SEG_PATH = 'input/segments.csv'  # 或 '/mnt/data_gen/segments_heavy.csv'
EDG_PATH = './input/edges.csv'
PHASE1   = 'lp'         # 'greedy' | 'relax' | 'lp'
SEED     = 2025
SLACK    = 1.08         # relax/lp 舍入时 z 的安全裕度

CLUSTER = [
    {'name': 'S1', 'S_C': 1.8, 'rho': 8.0, 'vgpu_weights': [0.30, 0.30, 0.25, 0.15]},
    {'name': 'S2', 'S_C': 1.6, 'rho':10.0, 'vgpu_weights': [0.25, 0.25, 0.25, 0.25]},
    {'name': 'S3', 'S_C': 2.0, 'rho': 6.0, 'vgpu_weights': [0.40, 0.30, 0.20, 0.10]},
    {'name': 'S4', 'S_C': 1.7, 'rho': 8.0, 'vgpu_weights': [0.35, 0.35, 0.20, 0.10]},
]

# --------------- Cluster utils ---------------
def enrich_cluster(cfg):
    cluster=[]
    for s in cfg:
        s=dict(s)
        s['S_G']   = s['rho'] * s['S_C']
        s['S_G_k'] = [w*s['S_G'] for w in s['vgpu_weights']]
        cluster.append(s)
    return cluster

def lower_bound_z(totals: pd.DataFrame, cluster: list[dict]) -> float:
    """三个下界取最大：全局平均；逐任务最短可行；以及0。"""
    SC_sum = sum(s['S_C'] for s in cluster)
    SG_sum = sum(s['S_G'] for s in cluster)
    lb_avg = max(totals['total_C'].sum()/SC_sum,
                 totals['total_G'].sum()/SG_sum)
    lb_jobs = 0.0
    for row in totals.itertuples():
        best = min(max(row.total_C/s['S_C'], row.total_G/s['S_G']) for s in cluster)
        lb_jobs = max(lb_jobs, best)
    return max(lb_avg, lb_jobs)

# --------------- Phase-1: greedy ---------------
def phase1_greedy(segments: pd.DataFrame, cluster: list[dict]):
    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    state = {s['name']: {'cpu_load':0.0,'gpu_load':0.0,'tasks':[]} for s in cluster}
    lookup = {s['name']: s for s in cluster}
    # GPU 重者先装，稳定 min–max
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

# --------------- Phase-1: relax (solver-free) ---------------
def phase1_relax(segments: pd.DataFrame, cluster: list[dict], seed=2025, slack=1.08):
    rng = np.random.RandomState(seed)
    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    z = lower_bound_z(totals, cluster) * slack

    # \hat{x}: 1 / max(C/S_C, G/S_G) 归一化成概率
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
            # 最小违约修复
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

# --------------- Phase-1: lp (PuLP) ---------------
def phase1_lp(segments: pd.DataFrame, cluster: list[dict], seed=2025, slack=1.05):
    try:
        import pulp
    except Exception:
        # 无求解器则退化到 relax
        return phase1_relax(segments, cluster, seed=seed, slack=slack)

    totals = segments.groupby('task_id').agg(total_C=('C_TFLOP','sum'),
                                             total_G=('G_TFLOP','sum')).reset_index()
    tasks = totals['task_id'].tolist()
    N = len(cluster)

    prob = pulp.LpProblem('phase1_lp', pulp.LpMinimize)
    x = {(i,n): pulp.LpVariable(f"x_{i}_{n}", lowBound=0, upBound=1) for i in tasks for n in range(N)}
    z = pulp.LpVariable("z", lowBound=0)
    prob += z

    # 指派约束
    for i in tasks:
        prob += pulp.lpSum(x[(i,n)] for n in range(N)) == 1

    # 负载约束
    Ti = totals.set_index('task_id')
    for n,srv in enumerate(cluster):
        prob += pulp.lpSum((Ti.loc[i,'total_C']/srv['S_C'])*x[(i,n)] for i in tasks) <= z
        prob += pulp.lpSum((Ti.loc[i,'total_G']/srv['S_G'])*x[(i,n)] for i in tasks) <= z

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    xhat = {i: np.array([pulp.value(x[(i,n)]) for n in range(N)], dtype=float) for i in tasks}
    zstar = float(pulp.value(z)) if pulp.value(z) is not None else lower_bound_z(totals, cluster)
    z_cap = zstar * slack

    # 随机化舍入（容量感知）
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
            # 最小违约修复
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

# --------------- Phase-2：Crit + CPU串行 + GPU-EFT ---------------
def compute_crit_for_task(task_id: str, segments: pd.DataFrame, edges: pd.DataFrame, srv: dict):
    segs = segments[segments['task_id']==task_id].copy()
    S_C = srv['S_C']; S_G_sum = sum(srv['S_G_k'])

    def w(row):
        return (row['C_TFLOP']/S_C) if row['type']=='CPU' else (row['G_TFLOP']/S_G_sum)
    segs['w'] = segs.apply(w, axis=1)

    task_edges = edges[edges['task_id']==task_id]
    succ = defaultdict(list); pred = defaultdict(list); nodes=set(segs['seg_id'].tolist())
    for _,e in task_edges.iterrows():
        succ[e['u']].append(e['v']); pred[e['v']].append(e['u'])
        nodes.add(int(e['u'])); nodes.add(int(e['v']))
    indeg = {v:0 for v in nodes}
    for v in nodes:
        for u in pred[v]: indeg[v]+=1

    q=deque([v for v in nodes if indeg[v]==0]); topo=[]
    while q:
        x=q.popleft(); topo.append(x)
        for y in succ[x]:
            indeg[y]-=1
            if indeg[y]==0: q.append(y)

    crit = {v:0.0 for v in nodes}
    w_map = {int(r.seg_id): float(r.w) for r in segs.itertuples()}
    for v in reversed(topo):
        crit[v] = w_map.get(v,0.0) + (max(crit[u] for u in succ[v]) if succ[v] else 0.0)
    return crit, succ, pred

def schedule_server(srv_name: str, segments: pd.DataFrame, edges: pd.DataFrame,
                    server_state: dict, cluster: list[dict]):
    srv = next(s for s in cluster if s['name']==srv_name)
    S_C = srv['S_C']; S_G_k = srv['S_G_k']
    T_cpu = 0.0; T_k = [0.0 for _ in S_G_k]
    finish = {}
    task_ids = server_state[srv_name]['tasks']

    # 预处理每个任务的图与 Crit
    task_struct = {}
    for tid in task_ids:
        crit, succ, pred = compute_crit_for_task(tid, segments, edges, srv)
        indeg = {v: len(pred[v]) for v in pred}
        for v in crit:
            indeg.setdefault(v,0); succ.setdefault(v,[]); pred.setdefault(v,[])
        task_struct[tid] = {'crit':crit,'succ':succ,'pred':pred,'indeg':indeg}

    # 初始就绪集
    ready=[]
    for tid in task_ids:
        st = task_struct[tid]
        for v,d in st['indeg'].items():
            if d==0:
                row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
                ready.append((tid,v,row['type']))

    # TODO 参数修改为全局变量，方便我做调整
    beta, betap = 0.0, 0.0  # 如需引入 CPU/GPU 惩罚可在此调整
    def prio(tid,v):
        st = task_struct[tid]
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        c = st['crit'][v]
        return c - beta*row['C_TFLOP'] if row['type']=='CPU' else c + betap*row['G_TFLOP']

    while ready:
        ready.sort(key=lambda x: prio(x[0],x[1]), reverse=True)
        tid,v,typ = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        if typ=='CPU':
            dur = row['C_TFLOP']/S_C
            start = T_cpu; end = start + dur; T_cpu = end
        else:
            preds = task_struct[tid]['pred'][v]
            rls = max([0.0]+[finish[(tid,u)] for u in preds])
            best_k, best_end = None, float('inf')
            for k,cap in enumerate(S_G_k):
                start_k = max(T_k[k], rls)
                end_k   = start_k + row['G_TFLOP']/cap
                if end_k < best_end: best_end, best_k = end_k, k
            T_k[best_k] = best_end; end = best_end
        finish[(tid,v)] = end
        for u in task_struct[tid]['succ'][v]:
            task_struct[tid]['indeg'][u] -= 1
            if task_struct[tid]['indeg'][u]==0:
                row2 = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]
                ready.append((tid,u,row2['type']))

    task_done={}
    for tid in task_ids:
        seg_ids = segments[segments['task_id']==tid]['seg_id'].tolist()
        task_done[tid] = max(finish[(tid,v)] for v in seg_ids)
    server_ms = max(task_done.values()) if task_done else 0.0
    return server_ms, task_done

# --------------- Driver ---------------
def run_scheduler(segments: pd.DataFrame, edges: pd.DataFrame, cluster_cfg: list[dict],
                  phase1: str = PHASE1, seed: int = SEED, slack: float = SLACK):
    cluster = enrich_cluster(cluster_cfg)
    if phase1 == 'greedy':
        server_state, z = phase1_greedy(segments, cluster)
    elif phase1 == 'relax':
        server_state, z = phase1_relax(segments, cluster, seed=seed, slack=slack)
    else:
        server_state, z = phase1_lp(segments, cluster, seed=seed, slack=slack)

    results={}
    for s in cluster:
        ms, td = schedule_server(s['name'], segments, edges, server_state, cluster)
        results[s['name']] = {'makespan': ms, 'tasks': td}

    batch_makespan = max(results[s]['makespan'] for s in results) if results else 0.0
    return batch_makespan, results, server_state, cluster, {'phase1': phase1, 'z': z}

def main():
    seg = pd.read_csv(SEG_PATH)
    edg = pd.read_csv(EDG_PATH)
    ms, results, state, cluster, info = run_scheduler(seg, edg, CLUSTER, phase1=PHASE1, seed=SEED, slack=SLACK)
    print(f"[Phase1={info['phase1']}] target z={info['z']:.6f}")
    print(f"Batch makespan (s): {ms:.6f}")
    for s in results:
        print(f"  {s}: makespan={results[s]['makespan']:.6f}, tasks={len(state[s]['tasks'])}")

if __name__ == '__main__':
    main()
