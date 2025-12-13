"""HEFT baseline (no communication cost).

This project models *zero communication overhead* across CPU/GPU and across servers.
Therefore, this implementation removes all comm-time terms and focuses strictly on
compute-time scheduling.

To align with your experiment design (two-stage: task packing -> per-server DAG scheduling):

Phase 1 (Task Packing, greedy):
  - Order tasks by their maximum upward-rank.
  - Assign each task to the server that minimizes the estimated completion level:
        max(cpu_load + C_i/S_C, gpu_load + G_i/S_G)

Phase 2 (Per-server HEFT list scheduling):
  - For each server, run list scheduling on the union of assigned tasks.
  - Node priority: upward-rank (global ranks computed on average resources).
  - Resource choice:
      * CPU node: single CPU timeline (serial)
      * GPU node: choose vGPU queue that yields the earliest finish time (EFT)

Return:
  overall_makespan, per_server_makespan
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from common import compute_global_ranks


def _task_totals(segments: pd.DataFrame) -> pd.DataFrame:
    return (
        segments.groupby("task_id")
        .agg(total_C=("C_TFLOP", "sum"), total_G=("G_TFLOP", "sum"))
        .reset_index()
    )


def _phase1_greedy_task_packing(
    segments: pd.DataFrame,
    cluster: List[Dict],
    ranks: Dict[Tuple[str, int], float],
) -> Dict[str, Dict]:
    """Greedy task-to-server assignment (no comm)."""
    totals = _task_totals(segments)

    # Task priority: maximum upward-rank among its nodes
    task_rank = {}
    for tid in totals["task_id"].tolist():
        sub = segments[segments["task_id"] == tid]
        task_rank[tid] = max((ranks.get((tid, int(v)), 0.0) for v in sub["seg_id"].tolist()), default=0.0)

    totals["task_rank"] = totals["task_id"].map(task_rank)
    totals = totals.sort_values(["task_rank", "total_G", "total_C"], ascending=[False, False, False])

    state = {s["name"]: {"cpu_load": 0.0, "gpu_load": 0.0, "tasks": []} for s in cluster}
    srv_by_name = {s["name"]: s for s in cluster}

    for r in totals.itertuples(index=False):
        tid = str(r.task_id)
        best_name = None
        best_metric = float("inf")
        for s in cluster:
            name = s["name"]
            # Use aggregated GPU capacity S_G (sum of vGPU queues) for packing estimate.
            cpu_t = float(r.total_C) / float(s["S_C"])
            gpu_t = float(r.total_G) / float(s["S_G"])
            metric = max(state[name]["cpu_load"] + cpu_t, state[name]["gpu_load"] + gpu_t)
            if metric < best_metric:
                best_metric = metric
                best_name = name

        st = state[best_name]
        srv = srv_by_name[best_name]
        st["cpu_load"] += float(r.total_C) / float(srv["S_C"])
        st["gpu_load"] += float(r.total_G) / float(srv["S_G"])
        st["tasks"].append(tid)

    return state


def _phase2_heft_on_server(
    server: Dict,
    segments: pd.DataFrame,
    edges: pd.DataFrame,
    ranks: Dict[Tuple[str, int], float],
    succ_all: Dict[str, Dict[int, List[int]]],
    pred_all: Dict[str, Dict[int, List[int]]],
    assigned_tasks: List[str],
) -> float:
    """List scheduling with priority=ranks and GPU choice by EFT (no comm)."""
    if not assigned_tasks:
        return 0.0

    S_C = float(server["S_C"])
    S_G_k = list(map(float, server["S_G_k"]))

    cpu_avail = 0.0
    gpu_avail = [0.0 for _ in S_G_k]
    finish: Dict[Tuple[str, int], float] = {}

    # Build per-node indegree for assigned tasks
    indeg: Dict[Tuple[str, int], int] = {}
    nodes: List[Tuple[str, int]] = []
    for tid in assigned_tasks:
        sub = segments[segments["task_id"] == tid]
        for v in sub["seg_id"].astype(int).tolist():
            indeg[(tid, v)] = len(pred_all.get(tid, {}).get(v, []))
            nodes.append((tid, v))

    ready = [n for n in nodes if indeg.get(n, 0) == 0]

    while ready:
        ready.sort(key=lambda tv: ranks.get(tv, 0.0), reverse=True)
        tid, v = ready.pop(0)
        row = segments[(segments["task_id"] == tid) & (segments["seg_id"] == v)].iloc[0]
        typ = str(row["type"]).upper()

        # release time from predecessors (no comm)
        rls = 0.0
        for u in pred_all.get(tid, {}).get(v, []):
            rls = max(rls, finish[(tid, int(u))])

        if typ == "CPU":
            start = max(cpu_avail, rls)
            dur = float(row["C_TFLOP"]) / S_C
            end = start + dur
            cpu_avail = end
        else:
            # GPU: choose queue that minimizes end time (EFT)
            best_end = float("inf")
            best_k = 0
            for k, cap in enumerate(S_G_k):
                start_k = max(gpu_avail[k], rls)
                dur = float(row["G_TFLOP"]) / float(cap)
                end_k = start_k + dur
                if end_k < best_end:
                    best_end = end_k
                    best_k = k
            gpu_avail[best_k] = best_end
            end = best_end

        finish[(tid, v)] = end

        for w in succ_all.get(tid, {}).get(v, []):
            key = (tid, int(w))
            if key in indeg:
                indeg[key] -= 1
                if indeg[key] == 0:
                    ready.append(key)

    return float(max([cpu_avail] + gpu_avail))


def run(
    segments: pd.DataFrame,
    edges: pd.DataFrame,
    cluster: List[Dict],
    *,
    seed: int = 2025,
    **_ignored,
) -> Tuple[float, Dict[str, float]]:
    """HEFT entrypoint (communication-free)."""
    # ranks/succ/pred are computed on average resources (classic HEFT rank definition)
    ranks, succ_all, pred_all = compute_global_ranks(segments, edges, cluster)

    server_state = _phase1_greedy_task_packing(segments, cluster, ranks)

    per: Dict[str, float] = {}
    for s in cluster:
        ms = _phase2_heft_on_server(
            s,
            segments,
            edges,
            ranks,
            succ_all,
            pred_all,
            assigned_tasks=server_state[s["name"]]["tasks"],
        )
        per[s["name"]] = float(ms)

    overall = max(per.values()) if per else 0.0
    return float(overall), per
