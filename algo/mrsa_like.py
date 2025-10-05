
"""
algo/mrsa_like.py
MRSA-like:
- Priority same as Hydra-like.
- Placement uses "FREE-first": pick resource with earliest availability after release, not EFT.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from common import schedule_global

BETA_C = 0.20
BETA_G = 0.40

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]):
    avg_cpu = float(np.mean([s['S_C'] for s in cluster]))
    all_q = []; [all_q.extend(s['S_G_k']) for s in cluster]
    avg_gpuq = float(np.mean(all_q)) if all_q else 1.0

    def task_priority(tid, v, typ, ctx):
        rank = ctx['ranks'][(tid, v)]
        row = ctx['segments'][(ctx['segments']['task_id']==tid)&(ctx['segments']['seg_id']==v)].iloc[0]
        if typ=='CPU':
            exp = float(row['C_TFLOP'])/avg_cpu
            return rank - BETA_C * exp
        else:
            exp = float(row['G_TFLOP'])/avg_gpuq
            return rank + BETA_G * exp
    return schedule_global(segments, edges, cluster, task_priority_fn=task_priority, selection_policy="FREE")
