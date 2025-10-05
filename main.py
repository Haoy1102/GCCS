
"""
main.py
Run all four algorithms from your IDE by pressing "Run" on main().
- Algorithms live under ./algo
- Shared utilities are in ./common.py
- Input CSVs are expected in ./input: segments.csv & edges.csv
- Outputs written to ./output/data
"""
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
    # 1) load data
    segments, edges = common.load_segments_edges()  # defaults to ./input/*
    # 2) make a default cluster (replace with your real cluster if you like)
    cluster = common.make_default_cluster(num_servers=6, rho=4.0, seed=2025)
    # 3) run all algorithms
    out_dir = Path("./output/data"); out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name, fn in ALGOS.items():
        ms, per_srv = fn(segments, edges, cluster)
        rows.append({"algorithm":name, "makespan":float(ms)})
        (out_dir / f"{name}.json").write_text(json.dumps(per_srv, indent=2, ensure_ascii=False))
        print(f"[{name}] makespan={ms:.6f}")
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote results to {out_dir}")

if __name__ == "__main__":
    main()
