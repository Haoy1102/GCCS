
"""
algo/hydra_like.py
Hydra-like (使用 c_n = w_n + e_n 作为优先级依据):
- 对每个就绪结点 n：计算其在所有资源上的 c_n，并取最小值作为该结点的“成本”
- 优先级 = - min c_n（越小越先）
- 放置策略仍用 EFT（与选到的 min c_n 资源一致）
"""
from __future__ import annotations
from typing import List, Dict
import pandas as pd
from common import schedule_global

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]):
    def task_priority(tid, v, typ, ctx):
        cpu_avail = ctx['cpu_avail']; gpu_avail = ctx['gpu_avail']
        release = float(ctx['release_time'])
        segs = ctx['segments']
        row = segs[(segs['task_id']==tid)&(segs['seg_id']==v)].iloc[0]

        best_c = float('inf')
        if typ=='CPU':
            for s in cluster:
                start = max(cpu_avail[s['name']], release)
                w = start - release
                e = float(row['C_TFLOP'])/s['S_C']
                c = w + e
                if c < best_c: best_c = c
        else:
            for s in cluster:
                for k,cap in enumerate(s['S_G_k']):
                    start = max(gpu_avail[s['name']][k], release)
                    w = start - release
                    e = float(row['G_TFLOP'])/cap
                    c = w + e
                    if c < best_c: best_c = c
        return -best_c
    return schedule_global(segments, edges, cluster, task_priority_fn=task_priority, selection_policy="EFT")
