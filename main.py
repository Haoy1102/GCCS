
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import common
from algo import two_phase as algo_two_phase
from algo import heft as algo_heft
from algo import hydra_like as algo_hydra
from algo import mrsa_like as algo_mrsa

ALGOS = {
    "GCCS-2Phase": algo_two_phase.run,
    "HEFT":        algo_heft.run,
    "HydraLike":   algo_hydra.run,
    "MRSALike":    algo_mrsa.run,
}

def main():
    segments, edges = common.load_segments_edges()
    # rho 自适应于本批 workload；kappa=4 作为常用值
    cluster = common.make_default_cluster(num_servers=6, rho='auto', kappa=4, segments=segments, seed=2025)

    out = Path("./output/data"); out.mkdir(parents=True, exist_ok=True)
    rows=[]

    # GCCS
    g_ms, g_per = ALGOS["GCCS-2Phase"](segments, edges, cluster)
    rows.append({"algorithm":"GCCS-2Phase","makespan":float(g_ms)})
    (out/"GCCS-2Phase.json").write_text(json.dumps(g_per, indent=2, ensure_ascii=False))
    print(f"[GCCS-2Phase] makespan={g_ms:.6f}")

    # HEFT（带通信开销，comm_scale 可视化调节 1.4~2.0）
    h_ms, h_per = ALGOS["HEFT"](segments, edges, cluster, comm_scale=1.6)
    rows.append({"algorithm":"HEFT","makespan":float(h_ms)})
    (out/"HEFT.json").write_text(json.dumps(h_per, indent=2, ensure_ascii=False))
    print(f"[HEFT] makespan={h_ms:.6f}")

    for name in ["HydraLike","MRSALike"]:
        ms, per = ALGOS[name](segments, edges, cluster)
        rows.append({"algorithm":name,"makespan":float(ms)})
        (out/f"{name}.json").write_text(json.dumps(per, indent=2, ensure_ascii=False))
        print(f"[{name}] makespan={ms:.6f}")

    pd.DataFrame(rows).to_csv(out/"summary.csv", index=False)
    print(f"Wrote results to {out}")

if __name__ == "__main__":
    main()
