#!/usr/bin/env python3
"""Macierz scenariuszy: konfiguracje bram x liczba osób -> jedna tabela porównawcza (results/matrix.csv).

  python run_matrix.py scenarios/juwenalia.yaml --n 3000 6000 10000 --workers 4
"""
import argparse, csv, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATE_SCENARIOS = {"wszystkie_otwarte": [], "Z1_zamknieta": ["Z1"], "Z2_zamknieta": ["Z2"], "Z1_i_Z2_zamkniete": ["Z1", "Z2"]}


def one(job):
    from evac.render import render_video
    from evac.sim import Simulation
    from evac.stats import write_outputs
    from run import build_cfg, resolve_map
    scen, n, name, closed, seed, outdir, backend, video = job
    a = argparse.Namespace(scenario=scen, n=n, seed=seed, closed=closed, tmax=None, party=None, set=[])
    cfg, _ = build_cfg(a)
    tag = f"N{n}_{name}_s{seed}"
    sim = Simulation(cfg, resolve_map(cfg, scen), backend=backend, record=video)
    res = sim.run()
    out = write_outputs(res, outdir, tag)
    if video:
        render_video(res, os.path.join(outdir, tag + ".mp4"), fps=int(cfg.output.fps))
    row = {"scenariusz_bram": name, **out["summary"]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--n", type=int, nargs="+", default=[3000, 6000, 10000])
    ap.add_argument("--gates", nargs="+", default=list(GATE_SCENARIOS), choices=list(GATE_SCENARIOS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--video", action="store_true", help="wideo dla każdego przebiegu (wolne)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.scenario, n, g, GATE_SCENARIOS[g], s, a.out, a.backend, a.video)
            for n in a.n for g in a.gates for s in a.seeds]
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, jobs))
    else:
        rows = []
        for j in jobs:
            print("->", j[1:5], flush=True)
            rows.append(one(j))
    path = os.path.join(a.out, "matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Zapisano", path)


if __name__ == "__main__":
    main()
