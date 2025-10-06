# algo/Hydra.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import defaultdict

# -----------------------
# Hydra: Global EFT with realistic comm/affinity/overheads
# -----------------------

def run(segments: pd.DataFrame,
        edges: pd.DataFrame,
        cluster: List[Dict],
        *,
        # ---- 通信建模 ----
        cross_comm_s: Optional[float] = None,   # 若提供则使用常数通信；None 则使用“大小感知”模型
        enable_cross_comm: bool = True,
        alpha: float = 0.02,                    # 大小感知通信中的固定开销 α (秒)
        gamma: float = 5.0,                     # 将 G_TFLOP 近似映射为数据量：size_mb ≈ gamma * G_TFLOP
        bw_default: float = 10000.0,            # 默认入向带宽 MB/s（若 cluster 里无 net_in_bw 字段）
        lambda_cong: float = 0.08,              # 并发惩罚系数；0 关闭拥塞效应
        # ---- 亲和与上下文 ----
        home_penalty: float = 0.03,             # 任务主机亲和软惩罚（落到非 home 机时在 release 上加）
        ctx_time: float = 0.004,                # vGPU 队列“上下文切换”开销；0 关闭
        cpu_quantum: float = 0.0,               # CPU 段统一微小调度粒度；0 关闭
        ) -> Tuple[float, Dict[str, float]]:
    """
    返回: (overall_makespan_seconds, per_server_makespan_dict)
    兼容旧调用: 不传 cross_comm_s 时仍可运行，此时使用“大小感知”通信模型。
    """

    segs = segments.copy()
    segs['type']   = segs['type'].astype(str).str.upper()
    segs['seg_id'] = segs['seg_id'].astype(int)

    e = edges.copy()
    e['u'] = e['u'].astype(int); e['v'] = e['v'].astype(int)

    # ----- 构图（每个任务内的 DAG）-----
    succ: Dict[Tuple[str,int], List[Tuple[str,int]]] = defaultdict(list)
    pred: Dict[Tuple[str,int], List[Tuple[str,int]]] = defaultdict(list)
    indeg: Dict[Tuple[str,int], int] = {}

    for tid in segs['task_id'].unique():
        ge = e[e['task_id'] == tid]
        nodes = segs[segs['task_id'] == tid]['seg_id'].astype(int).tolist()
        for _, r in ge.iterrows():
            u = int(r['u']); v = int(r['v'])
            succ[(tid,u)].append((tid,v))
            pred[(tid,v)].append((tid,u))
        for v in nodes:
            indeg.setdefault((tid,v), 0)
        # 仅累计本任务内部入度
        for (t,u), lst in list(succ.items()):
            if t != tid:
                continue
            for (t2,v) in lst:
                if t2 == tid:
                    indeg[(tid,v)] = indeg.get((tid,v), 0) + 1

    # ----- 初始就绪 -----
    ready: List[Tuple[str,int,str]] = []
    for (tid,v), d in indeg.items():
        if d == 0:
            typ = str(segs[(segs['task_id']==tid) & (segs['seg_id']==v)].iloc[0]['type']).upper()
            ready.append((tid, v, typ))

    # ----- 服务器状态 -----
    state = {s['name']: {
                'T_cpu': 0.0,
                'T_k': [0.0 for _ in s['S_G_k']],
                'last_task_on_queue': [None for _ in s['S_G_k']]
            } for s in cluster}
    cluster_by_name = {s['name']: s for s in cluster}

    # 段完成时刻与所在服务器（用于通信判断）
    finish: Dict[Tuple[str,int], float] = {}
    placed_srv: Dict[Tuple[str,int], str] = {}

    # 任务的“主机”（首段落地处）
    task_home: Dict[str, str] = {}

    # 简易“并发通信”窗口集合：元素为 (start, end)
    active_comm: List[Tuple[float,float]] = []

    # ---- 工具函数 ----
    def _concurrency_at(t: float) -> int:
        # 统计时刻 t 有多少通信窗口覆盖
        cnt = 0
        for (a,b) in active_comm:
            if a <= t < b:
                cnt += 1
        return cnt

    def _edge_size_mb_for_seg(row: pd.Series) -> float:
        # 若无真实数据，近似地使用该段 G_TFLOP 估算数据量
        g = float(row.get('G_TFLOP', 0.0))
        return max(1.0, gamma * g)

    def _comm_extra(base_time: float, from_srv: str, to_srv: str, row: pd.Series) -> float:
        """返回从 from_srv -> to_srv 的额外通信时延（秒），按配置选择常数或大小感知模型，并附加拥塞因子。"""
        if (not enable_cross_comm) or (from_srv == to_srv):
            return 0.0
        # 常数模型优先
        if cross_comm_s is not None:
            extra = float(cross_comm_s)
        else:
            # 大小 + 带宽 + 拥塞
            size_mb = _edge_size_mb_for_seg(row)
            bw = float(cluster_by_name.get(to_srv, {}).get('net_in_bw', bw_default))
            base = alpha + size_mb / max(1e-9, bw)
            if lambda_cong > 0.0:
                base *= (1.0 + lambda_cong * _concurrency_at(base_time))
            extra = base
        return extra

    # ----- 调度主循环：每步从 ready 中选择(任务段, 服务器/队列)使 EFT 最小 -----
    while ready:
        best = None   # (eft, cn, tid, v, typ, sname, dev, k, start, end)

        for tid, v, typ in ready:
            row = segs[(segs['task_id']==tid) & (segs['seg_id']==v)].iloc[0]
            is_cpu = (typ == 'CPU')

            for s in cluster:
                sname = s['name']

                # 计算带通信的 release（多前驱取 max(base + extra)）
                if pred.get((tid, v)):
                    release = 0.0
                    for (ptid, pu) in pred[(tid, v)]:
                        base = finish[(ptid, pu)]
                        from_srv = placed_srv[(ptid, pu)]
                        extra = _comm_extra(base, from_srv, sname, row)
                        release = max(release, base + extra)
                else:
                    release = 0.0

                # 亲和性（soft）：若已有 home，且这次放到非 home 机，增加 release
                home = task_home.get(tid)
                if (home is not None) and (sname != home):
                    release += home_penalty

                if is_cpu:
                    start = max(state[sname]['T_cpu'], release)
                    if cpu_quantum > 0.0:
                        start += cpu_quantum
                    exec_t = float(row['C_TFLOP']) / float(s['S_C'])
                    eft    = start + exec_t
                    cn     = (start - release) + exec_t
                    cand   = (eft, cn, tid, v, typ, sname, 'CPU', None, start, eft)
                    if (best is None) or (cand < best):
                        best = cand
                else:
                    gflop = float(row['G_TFLOP'])
                    for k, cap in enumerate(s['S_G_k']):
                        start_k = max(state[sname]['T_k'][k], release)
                        # vGPU 队列“上下文切换”开销：如果换了任务，则加一点启动时间
                        if ctx_time > 0.0 and state[sname]['last_task_on_queue'][k] != tid:
                            start_k += ctx_time
                        exec_t  = gflop / float(cap)
                        eft_k   = start_k + exec_t
                        cn_k    = (start_k - release) + exec_t
                        cand    = (eft_k, cn_k, tid, v, typ, sname, 'GPU', k, start_k, eft_k)
                        if (best is None) or (cand < best):
                            best = cand

        # ---- 执行最优候选 ----
        _, _, tid, v, typ, sname, dev, k, start, end = best
        if dev == 'CPU':
            state[sname]['T_cpu'] = end
        else:
            state[sname]['T_k'][k] = end
            state[sname]['last_task_on_queue'][k] = tid

        # 完成登记
        finish[(tid, v)] = end
        placed_srv[(tid, v)] = sname

        # 若该段有跨服前驱，登记通信窗口（用于拥塞估计）
        if pred.get((tid, v)):
            row = segs[(segs['task_id']==tid) & (segs['seg_id']==v)].iloc[0]
            for (ptid, pu) in pred[(tid, v)]:
                from_srv = placed_srv[(ptid, pu)]
                if enable_cross_comm and (from_srv != sname):
                    base = finish[(ptid, pu)]
                    extra = _comm_extra(base, from_srv, sname, row)
                    if extra > 0.0:
                        active_comm.append((base, base + extra))

        # 设定/保持 task home
        if tid not in task_home:
            task_home[tid] = sname

        # ---- 更新就绪 ----
        # 移除刚完成的
        ready = [(tt,vv,ttyp) for (tt,vv,ttyp) in ready if not (tt==tid and vv==v)]
        # 处理后继
        for (t2, v2) in succ.get((tid, v), []):
            indeg[(t2, v2)] -= 1
            if indeg[(t2, v2)] == 0:
                ttyp = str(segs[(segs['task_id']==t2) & (segs['seg_id']==v2)].iloc[0]['type']).upper()
                ready.append((t2, v2, ttyp))

    # ----- 汇总 -----
    per_srv = {s['name']: 0.0 for s in cluster}
    for (tid, v), t in finish.items():
        sname = placed_srv[(tid, v)]
        per_srv[sname] = max(per_srv[sname], t)
    overall = max(per_srv.values()) if per_srv else 0.0
    return float(overall), per_srv
