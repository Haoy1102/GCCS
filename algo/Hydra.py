# algo/Hydra.py
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
from common import compute_global_ranks


def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict], **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Hydra Revised: Efficiency-First + Data-Locality Sticky

    核心逻辑：
    1. Phase 1: 任务分配 (GPU 偏好)
       - 优先把任务分给"能提供最强 GPU 算力"的服务器。
       - 对 CPU 负载不敏感 (认为 CPU 总是够用的)。

    2. Phase 2: 单机调度 (Rank + Sticky CPU)
       - 排序：使用 Rank (聪明的)。
       - 选卡：GPU 选 EFT 最小的 (聪明的)。
       - CPU 调度 (缺陷点)：Data Locality Sticky。
         CPU 任务必须在"数据所在地"运行。在单机场景下，这意味着它只能排队。
         Hydra 不会为了 CPU 负载均衡而尝试把子任务"偷"到空闲核上（本模型也不支持任务内拆分，所以这个设定是合理的）。
         但它的缺陷在于 Phase 1 分配时忽视了 CPU 负载，导致某些机器 CPU 积压严重。
    """

    srv_names = [s['name'] for s in cluster]
    srv_dict = {s['name']: s for s in cluster}
    # compute_global_ranks 返回的是：
    #   ranks[(task_id, seg_id)] = upward-rank
    #   succ_all[task_id][u] = [v...]
    #   pred_all[task_id][v] = [u...]
    ranks, succ_all, pred_all = compute_global_ranks(segments, edges, cluster)

    # === Phase 1: 任务分配 (GPU-Centric Greedy) ===
    task_info = []
    all_tasks = segments['task_id'].unique()
    for tid in all_tasks:
        sub = segments[segments['task_id'] == tid]
        g_load = sub[sub['type'].str.upper() == 'GPU']['G_TFLOP'].sum()
        max_rank = max(ranks.get((tid, vid), 0.0) for vid in sub['seg_id'])
        task_info.append((tid, max_rank, g_load))

    task_info.sort(key=lambda x: x[1], reverse=True)

    # 水位记录
    # Hydra 只关心 GPU 是否高效，它倾向于把重 GPU 任务给 A100
    srv_gpu_load = {name: [0.0] * len(srv_dict[name]['S_G_k']) for name in srv_names}
    task_map = {}

    for tid, _, g_req in task_info:
        best_srv = None
        min_score = float('inf')

        for sname in srv_names:
            s = srv_dict[sname]
            # 评分标准：GPU 预计完成时间
            # Hydra 极其讨厌把任务给弱卡，所以它倾向于塞给强卡，哪怕强卡已经有点忙

            # 找到该服务器上"相对空闲"且"算力强"的卡
            # 简化模型：假设分给该服务器上算力最强的那张卡 (Efficiency First)
            # 或者平均分配

            # 这里模拟 Hydra 的偏好：Workload / Max_Capacity
            # 它希望 execution_time 最小
            max_cap = max(s['S_G_k']) if s['S_G_k'] else 1.0
            exec_time = g_req / max_cap

            # 加上当前的平均排队
            avg_queue = sum(srv_gpu_load[sname]) / len(srv_gpu_load[sname]) if srv_gpu_load[sname] else 0.0

            eft = avg_queue + exec_time

            if eft < min_score:
                min_score = eft
                best_srv = sname

        task_map[tid] = best_srv

        # 更新水位 (Hydra 意识不到 CPU 也在涨)
        s = srv_dict[best_srv]
        avg_cap = sum(s['S_G_k']) / len(s['S_G_k']) if s['S_G_k'] else 1.0
        inc = g_req / avg_cap
        for k in range(len(srv_gpu_load[best_srv])):
            srv_gpu_load[best_srv][k] += inc

    # === Phase 2: 真实执行 (Local) ===
    # 这里如实模拟 CPU 和 GPU

    final_srv_makespan = {}

    for sname in srv_names:
        my_tasks = [t for t, s in task_map.items() if s == sname]
        if not my_tasks:
            final_srv_makespan[sname] = 0.0
            continue

        s_info = srv_dict[sname]
        cpu_avail = 0.0
        gpu_avail = [0.0] * len(s_info['S_G_k'])
        finished = {}

        # 本机 Ready Queue
        indeg = {}
        nodes = set()
        for tid in my_tasks:
            t_segs = segments[segments['task_id'] == tid]
            for _, row in t_segs.iterrows():
                vid = row['seg_id']
                indeg[(tid, vid)] = len(pred_all.get(tid, {}).get(vid, []))
                nodes.add((tid, vid))
        ready = [n for n in nodes if indeg.get(n, 0) == 0]

        while ready:
            # 排序：按 Rank (Hydra 是聪明的)
            ready.sort(key=lambda x: ranks.get(x, 0.0), reverse=True)
            tid, vid = ready.pop(0)

            row = segments[(segments['task_id'] == tid) & (segments['seg_id'] == vid)].iloc[0]
            stype = str(row['type']).upper()

            ready_time = 0.0
            for pu in pred_all.get(tid, {}).get(vid, []):
                ready_time = max(ready_time, finished.get((tid, pu), 0.0))

            if stype == 'CPU':
                # CPU 照常排队
                # 但因为 Phase 1 没考虑 CPU 均衡，这里可能会发生拥塞
                start = max(cpu_avail, ready_time)
                dur = float(row['C_TFLOP']) / s_info['S_C']
                end = start + dur
                cpu_avail = end
                finished[(tid, vid)] = end
            else:
                # GPU: 选 EFT 最小
                best_k = 0
                best_end = float('inf')
                for k, cap in enumerate(s_info['S_G_k']):
                    start = max(gpu_avail[k], ready_time)
                    dur = float(row['G_TFLOP']) / cap
                    end = start + dur
                    if end < best_end:
                        best_end = end
                        best_k = k

                gpu_avail[best_k] = best_end
                finished[(tid, vid)] = best_end

            for su in succ_all.get(tid, {}).get(vid, []):
                key = (tid, su)
                if key in indeg:
                    indeg[key] -= 1
                    if indeg[key] == 0:
                        ready.append(key)

        final_srv_makespan[sname] = max(cpu_avail, max(gpu_avail) if gpu_avail else 0.0)

    return max(final_srv_makespan.values()), final_srv_makespan