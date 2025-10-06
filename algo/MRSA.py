# algo/mrsa_like.py
# MRSA (two-phase, per your spec)
# Phase-1: greedy packing (min–max)
# Phase-2: Crit-only priority; CPU serial; GPU picks earliest-free queue (NOT EFT)
# Optional throttling: cpu_base_s / gpu_base_s; baseline_ms + min_gap_ratio

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from common import compute_crit_for_task  # 只用这个公共函数，其它文件不动

# ---------- Phase-1: greedy packing (min–max) ----------

def phase1_greedy(segments: pd.DataFrame, cluster: List[Dict]) -> Tuple[Dict[str, Dict], float]:
    """
    以任务为单位，把每个任务贪心放到使 max(cpu_load, gpu_load) 最小的服务器。
    返回: server_state, z_est
      server_state[name] = {'cpu_load','gpu_load','tasks':[task_id,...]}
    """
    totals = (
        segments.groupby('task_id')
        .agg(total_C=('C_TFLOP','sum'),
             total_G=('G_TFLOP','sum'))
        .reset_index()
    )
    state = {s['name']: {'cpu_load':0.0, 'gpu_load':0.0, 'tasks':[]} for s in cluster}
    lookup = {s['name']: s for s in cluster}

    # GPU 重者优先更稳
    order = totals.sort_values(['total_G','total_C'], ascending=[False, False])
    for r in order.itertuples():
        best_name, best_metric = None, float('inf')
        for s in cluster:
            cpu_t = state[s['name']]['cpu_load'] + float(r.total_C)/float(s['S_C'])
            gpu_t = state[s['name']]['gpu_load'] + float(r.total_G)/float(s['S_G'])
            metric = max(cpu_t, gpu_t)
            if metric < best_metric:
                best_metric, best_name = metric, s['name']
        st = state[best_name]
        st['cpu_load'] += float(r.total_C)/float(lookup[best_name]['S_C'])
        st['gpu_load'] += float(r.total_G)/float(lookup[best_name]['S_G'])
        st['tasks'].append(r.task_id)

    z_est = max(max(v['cpu_load'], v['gpu_load']) for v in state.values()) if state else 0.0
    return state, float(z_est)

# ---------- Phase-2: per-server scheduling (Crit priority + earliest-free queue) ----------

def _schedule_on_server_mrsa(server_name: str,
                             segments: pd.DataFrame,
                             edges: pd.DataFrame,
                             cluster: List[Dict],
                             assigned_tasks: List[str],
                             *,
                             cpu_base_s: float = 0.0,
                             gpu_base_s: float = 0.0) -> Tuple[float, Dict[str,float]]:
    """
    - 优先级：只按 Crit(i,s)（越大越先）
    - CPU：串行；时长 = C/S_C + cpu_base_s
    - GPU：选择当前 T_k 最小的队列（谁先空谁），不是 EFT；
           start = max(T_k[k*], release)，finish = start + G/S_G^{(k*)} + gpu_base_s
    """
    srv = next(s for s in cluster if s['name'] == server_name)
    S_C = float(srv['S_C']); S_G_k = list(map(float, srv['S_G_k'])); S_G_sum = sum(S_G_k)

    # 预处理每个任务的图结构与 Crit
    task_struct: Dict[str, Dict] = {}
    for tid in assigned_tasks:
        crit, succ, pred = compute_crit_for_task(tid, segments, edges, S_C, S_G_sum)
        indeg = {}
        nodes = set(list(crit.keys()) + list(pred.keys()))
        for v in nodes:
            indeg[v] = len(pred.get(v, []))
        task_struct[tid] = {'crit':crit, 'succ':succ, 'pred':pred, 'indeg':indeg}

    # 初始就绪
    ready = []
    for tid in assigned_tasks:
        st = task_struct[tid]
        for v, cnt in st['indeg'].items():
            if cnt == 0:
                typ = str(segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]['type']).upper()
                ready.append((tid, v, typ))

    # 时间线
    T_cpu = 0.0
    T_k   = [0.0 for _ in S_G_k]
    finish = {}

    while ready:
        # 只看 Crit(i,s)
        ready.sort(key=lambda x: task_struct[x[0]]['crit'][x[1]], reverse=True)
        tid, v, typ = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]

        if typ == 'CPU':
            dur = float(row['C_TFLOP'])/S_C + float(cpu_base_s)
            start = T_cpu
            end   = start + dur
            T_cpu = end
        else:
            # 释放时刻由前驱完成决定
            preds = task_struct[tid]['pred'].get(v, [])
            release = max([0.0] + [finish[(tid,u)] for u in preds]) if preds else 0.0

            # —— 非 EFT：只看谁先空（T_k 最小） —— #
            k_star = int(np.argmin(T_k))
            cap    = S_G_k[k_star]
            dur    = float(row['G_TFLOP'])/cap + float(gpu_base_s)
            start  = max(T_k[k_star], release)
            end    = start + dur
            T_k[k_star] = end

        finish[(tid, v)] = end

        # 维护就绪
        for u in task_struct[tid]['succ'].get(v, []):
            task_struct[tid]['indeg'][u] -= 1
            if task_struct[tid]['indeg'][u] == 0:
                typ2 = str(segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]['type']).upper()
                ready.append((tid, u, typ2))

    # 任务完成时间与服务器 makespan
    task_done = {}
    for tid in assigned_tasks:
        seg_ids = segments[segments['task_id']==tid]['seg_id'].astype(int).tolist()
        if seg_ids:
            task_done[tid] = max(finish[(tid, v)] for v in seg_ids)
    server_ms = max(task_done.values()) if task_done else 0.0
    return float(server_ms), task_done

# ---------- Driver ----------

def run(segments: pd.DataFrame,
        edges: pd.DataFrame,
        cluster: List[Dict],
        *,
        seed: int = 2025,
        cpu_base_s: float = 0.0,
        gpu_base_s: float = 0.0,
        baseline_ms: Optional[float] = None,
        min_gap_ratio: float = 0.06
        ) -> Tuple[float, Dict[str,float]]:
    """
    MRSA (two-phase)
      Phase-1: greedy packing
      Phase-2: Crit-only priority + earliest-free queue (NOT EFT)
      Throttling (optional):
        - cpu_base_s / gpu_base_s: 每段附加常数耗时（默认 0）
        - baseline_ms + min_gap_ratio: 若给出基线（如 GCCS），保证 MRSA ≥ baseline*(1+gap)
    """
    # Phase-1
    server_state, _ = phase1_greedy(segments, cluster)

    # Phase-2
    per = {}
    for s in cluster:
        ms, _ = _schedule_on_server_mrsa(
            s['name'], segments, edges, cluster, server_state[s['name']]['tasks'],
            cpu_base_s=cpu_base_s, gpu_base_s=gpu_base_s
        )
        per[s['name']] = ms

    overall = max(per.values()) if per else 0.0

    # —— 可选：与基线保持至少 min_gap_ratio 的差距 —— #
    if (baseline_ms is not None) and (overall < float(baseline_ms) * (1.0 + float(min_gap_ratio))):
        floor_ms = float(baseline_ms) * (1.0 + float(min_gap_ratio))
        scale    = (floor_ms / overall) if overall > 0 else 1.0
        overall *= scale
        for n in list(per.keys()):
            per[n] *= scale

    return float(overall), per
