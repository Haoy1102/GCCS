
"""
Run a grid over rho in {0.5R, R, 2R} and kappa in {2..6}.
Write CSV: rho,kappa,method,makespan  to ./output/data/sweep.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import common
from algo import two_phase as algo_two_phase
from algo import heft as algo_heft
from algo import hydra_like as algo_hydra
from algo import mrsa_like as algo_mrsa

def main():
    seg, edg = common.load_segments_edges()
    R = common.workload_ratio_R(seg)
    rho_list = [0.5*R, 1.0*R, 2.0*R]  # 可扩展到 4R
    kappas = [2,3,4,5,6]
    rows=[]
    for rho in rho_list:
        for kappa in kappas:
            cluster = common.make_default_cluster(num_servers=6, rho=rho, kappa=kappa, segments=seg, seed=2025)
            g_ms,_ = algo_two_phase.run(seg, edg, cluster)
            h_ms,_ = algo_heft.run(seg, edg, cluster, comm_scale=1.6)
            y_ms,_ = algo_hydra.run(seg, edg, cluster)
            m_ms,_ = algo_mrsa.run(seg, edg, cluster)
            rows += [
                {"rho":round(rho,3),"kappa":kappa,"method":"GCCS","makespan":float(g_ms)},
                {"rho":round(rho,3),"kappa":kappa,"method":"HEFT","makespan":float(h_ms)},
                {"rho":round(rho,3),"kappa":kappa,"method":"Hydra","makespan":float(y_ms)},
                {"rho":round(rho,3),"kappa":kappa,"method":"MRSA","makespan":float(m_ms)},
            ]
            print(f"rho={rho:.3f}, kappa={kappa} done.")
    out = Path("./output/data"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"sweep.csv", index=False)
    print(f"Wrote {out/'sweep.csv'}")

if __name__ == "__main__":
    main()
