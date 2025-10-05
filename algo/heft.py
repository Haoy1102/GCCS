
"""
algo/heft.py
Global HEFT-style baseline:
- Ordering by global upward rank (average speeds)
- For each ready node, select the CPU/vGPU resource that minimizes EFT across all servers.
- No packing stage. No communication cost.
"""
from __future__ import annotations
from typing import List, Dict
import pandas as pd
from common import schedule_global

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict]):
    def task_priority(tid, v, typ, ctx):
        return ctx['ranks'][(tid, v)]  # pure rank2
    return schedule_global(segments, edges, cluster, task_priority_fn=task_priority, selection_policy="EFT")
