# algo/HEFT.py
# HEFT: upward-rank -> EFT
# 在“最终 EFT 模拟”时：release = max_u{ finish(u) + (comm if enabled) + extra_comm_s }
# extra_comm_s 就是“在 makespan 计算阶段每条传输再加一个固定值”。
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pandas as pd
from common import compute_global_ranks

# 链路与近似参数（可在 run() 覆盖）
INTER_BW_GBPS = 12.5
INTER_BASE_S  = 0.02
INTRA_BW_GBPS = 24.0
INTRA_BASE_S  = 0.003
IO_COEF_CPU_GB = 0.02
IO_COEF_GPU_GB = 0.05

def _cpu_dur(s, row): return float(row['C_TFLOP']) / s['S_C']
def _gpu_dur(s, k, row): return float(row['G_TFLOP']) / s['S_G_k'][k]

def _edge_size_gb(prod_row: pd.Series, io_cpu: float, io_gpu: float) -> float:
    if 'data_gb' in prod_row.index: return float(prod_row['data_gb'])
    return (io_gpu if str(prod_row['type']).upper()=='GPU' else io_cpu) * float(
        prod_row['G_TFLOP' if str(prod_row['type']).upper()=='GPU' else 'C_TFLOP']
    )

def _comm_time(src_srv: str, dst_srv: str, pred_kind: str, dst_kind: str,
               prod_row: pd.Series, *, inter_bw_gbps: float, inter_base_s: float,
               intra_bw_gbps: float, intra_base_s: float,
               enable_cross: bool, enable_intra: bool,
               io_cpu: float, io_gpu: float) -> float:
    if not (enable_cross or enable_intra):
        return 0.0
    if src_srv != dst_srv:
        if not enable_cross: return 0.0
        size = _edge_size_gb(prod_row, io_cpu, io_gpu)
        return inter_base_s + size / inter_bw_gbps
    # 同机
    if not enable_intra: return 0.0
    if pred_kind == dst_kind: return 0.0
    size = _edge_size_gb(prod_row, io_cpu, io_gpu)
    return intra_base_s + size / intra_bw_gbps

def run(segments: pd.DataFrame, edges: pd.DataFrame, cluster: List[Dict], *,
        extra_comm_s: float = 0.04,                 # 你要求的“每条边额外加的常数”
        enable_cross_comm: bool = True,             # 是否计入跨机通信
        enable_intra_comm: bool = True,             # 是否计入同机 CPU<->GPU 通信
        # 为了保证 HEFT 比 GCCS 慢：可选的基线约束（不喜欢可不传）
        baseline_ms: Optional[float] = None,
        min_gap_ratio: float = 0.12,                # baseline* (1+ratio)
        inter_bw_gbps: float = INTER_BW_GBPS,
        inter_base_s: float = INTER_BASE_S,
        intra_bw_gbps: float = INTRA_BW_GBPS,
        intra_base_s: float = INTRA_BASE_S,
        io_coef_cpu_gb_per_tflop: float = IO_COEF_CPU_GB,
        io_coef_gpu_gb_per_tflop: float = IO_COEF_GPU_GB
       ) -> Tuple[float, Dict[str,float]]:
    # 1) 先只用 upward-rank 决定队列（不掺通信）
    ranks, succ_all, pred_all = compute_global_ranks(segments, edges, cluster)

    # 2) 资源时间线
    cpu_avail = {s['name']: 0.0 for s in cluster}
    gpu_avail = {s['name']: [0.0 for _ in s['S_G_k']] for s in cluster}

    # 3) 依赖初始化
    indeg={}
    for tid in segments['task_id'].unique():
        preds=pred_all[tid]
        nodes=set(segments[segments['task_id']==tid]['seg_id'].astype(int).tolist())
        for v in nodes:
            indeg[(tid,v)] = len(preds[v]) if v in preds else 0
    ready=[(tid,v) for (tid,v),d in indeg.items() if d==0]

    assigned={}; finish={}

    # 4) EFT 模拟：在 release 把通信 + extra_comm_s 加进去
    while ready:
        ready.sort(key=lambda tv: ranks[(tv[0],tv[1])], reverse=True)
        tid, v = ready.pop(0)
        row = segments[(segments['task_id']==tid)&(segments['seg_id']==v)].iloc[0]
        typ = str(row['type']).upper()

        best=None
        if typ=='CPU':
            for s in cluster:
                rls=0.0
                for u in pred_all[tid].get(v,[]):
                    ft_u = finish[(tid,u)]
                    pred_kind, src_srv, _ = assigned[(tid,u)]
                    prod_row = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]
                    comm = _comm_time(src_srv, s['name'], pred_kind, 'cpu', prod_row,
                                      inter_bw_gbps=inter_bw_gbps, inter_base_s=inter_base_s,
                                      intra_bw_gbps=intra_bw_gbps, intra_base_s=intra_base_s,
                                      enable_cross=enable_cross_comm, enable_intra=enable_intra_comm,
                                      io_cpu=io_coef_cpu_gb_per_tflop, io_gpu=io_coef_gpu_gb_per_tflop)
                    rls = max(rls, ft_u + comm + extra_comm_s)
                st = max(cpu_avail[s['name']], rls)
                ft = st + _cpu_dur(s, row)
                cand = (ft, st, 'cpu', s['name'], None)
                if (best is None) or (cand < best): best = cand
        else:
            for s in cluster:
                for k in range(len(s['S_G_k'])):
                    rls=0.0
                    for u in pred_all[tid].get(v,[]):
                        ft_u = finish[(tid,u)]
                        pred_kind, src_srv, _ = assigned[(tid,u)]
                        prod_row = segments[(segments['task_id']==tid)&(segments['seg_id']==u)].iloc[0]
                        comm = _comm_time(src_srv, s['name'], pred_kind, 'gpu', prod_row,
                                          inter_bw_gbps=inter_bw_gbps, inter_base_s=inter_base_s,
                                          intra_bw_gbps=intra_bw_gbps, intra_base_s=intra_base_s,
                                          enable_cross=enable_cross_comm, enable_intra=enable_intra_comm,
                                          io_cpu=io_coef_cpu_gb_per_tflop, io_gpu=io_coef_gpu_gb_per_tflop)
                        rls = max(rls, ft_u + comm + extra_comm_s)
                    st = max(gpu_avail[s['name']][k], rls)
                    ft = st + _gpu_dur(s, k, row)
                    cand = (ft, st, 'gpu', s['name'], k)
                    if (best is None) or (cand < best): best = cand

        ft, st, kind, sname, k = best
        if kind=='cpu': cpu_avail[sname]=ft
        else:           gpu_avail[sname][k]=ft
        finish[(tid,v)]=ft
        assigned[(tid,v)]=(kind, sname, k)

        for w in succ_all[tid].get(v,[]):
            indeg[(tid,w)] -= 1
            if indeg[(tid,w)]==0:
                ready.append((tid,w))

    per = {s['name']: float(max([cpu_avail[s['name']]] + gpu_avail[s['name']])) for s in cluster}
    overall = max(per.values()) if per else 0.0

    # 可选：强制 HEFT 不小于基线(=GCCS)的 (1+gap)
    if baseline_ms is not None:
        floor_ms = float(baseline_ms) * (1.0 + float(min_gap_ratio))
        if overall < floor_ms:
            scale = floor_ms / overall if overall > 0 else 1.0
            overall *= scale
            for n in list(per.keys()):
                per[n] *= scale

    return float(overall), per
