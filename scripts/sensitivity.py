#!/usr/bin/env python3
"""Analiza wrażliwości "jeden parametr naraz": który parametr najmocniej zmienia wynik ewakuacji.

Dla każdego parametru z listy PARAMS puszcza przebiegi z wartością niską i wysoką (reszta jak w scenariuszu),
po kilka ziaren, i zapisuje zmianę T90, liczby rannych i odsetka ewakuowanych względem przebiegu bazowego.
Przydaje się do wskazania, które wartości "wstępne" warto skalibrować w pierwszej kolejności.

  python scripts/sensitivity.py scenarios/etap2.yaml --n 1500 --seeds 1 2 3
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from evac.config import load_config  # noqa: E402
from evac.sim import Simulation  # noqa: E402
from evac.stats import summarize  # noqa: E402
from run import resolve_map  # noqa: E402

# parametr: (wartość niska, wartość wysoka)
PARAMS = {
    "behaviour.v_max_mps": (1.6, 2.4),
    "alarm.info_speed_mps": (1.5, 6.0),
    "alarm.wave_range_m": (30.0, 120.0),
    "alarm.global_delay_s": ([30, 30], [180, 180]),
    "behaviour.gawkers.share": (0.0, 0.15),
    "behaviour.ignore_alarm_share": (0.05, 0.40),
    "behaviour.contagion.rate_per_s": (0.05, 0.30),
    "exits.choice.temperature_m": (4.0, 30.0),
    "exits.choice.knowledge.W6": (0.05, 0.60),
    "exits.choice.width_bonus_m": (0.0, 20.0),
    "injury.dose_severe_Nspm": (6000.0, 24000.0),
    "hazard.wind.speed_mps": (0.0, 2.0),
}
OUT = ["T90_s", "odsetek_ewakuowanych_%", "poszkodowani_razem", "nie_wyszli_ciezko_ranni"]


def run(scen, n, seed, sets, variant, backend):
    cfg, _ = load_config(scen, {"agents": {"count": n}, "seed": seed, "output": {"heatmap": False}}, sets,
                         variant=variant)
    res = Simulation(cfg, resolve_map(cfg, scen), backend=backend, record=False).run()
    s = summarize(res)
    return [float(s[k]) if s[k] != "" else np.nan for k in OUT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="tylko wybrane parametry")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "sensitivity.csv"))
    a = ap.parse_args()

    def mean(sets):
        return np.nanmean([run(a.scenario, a.n, s, sets, a.variant, a.backend) for s in a.seeds], axis=0)

    base = mean([])
    print("baza: " + ", ".join(f"{k}={v:.1f}" for k, v in zip(OUT, base)), flush=True)
    rows = []
    for key, (lo, hi) in PARAMS.items():
        if a.only and key not in a.only:
            continue
        r_lo, r_hi = mean([f"{key}={lo}"]), mean([f"{key}={hi}"])
        row = {"parametr": key, "niska": lo, "wysoka": hi}
        for k, b, x, y in zip(OUT, base, r_lo, r_hi):
            row[f"{k}_baza"] = round(float(b), 1)
            row[f"{k}_niska"] = round(float(x), 1)
            row[f"{k}_wysoka"] = round(float(y), 1)
        row["rozrzut_T90_s"] = round(float(abs(r_hi[0] - r_lo[0])), 1)
        rows.append(row)
        print(f"{key:36s} T90: {r_lo[0]:6.1f} .. {r_hi[0]:6.1f}   ranni: {r_lo[2]:5.1f} .. {r_hi[2]:5.1f}", flush=True)
    rows.sort(key=lambda r: -r["rozrzut_T90_s"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Zapisano", a.out, "(posortowane wg wpływu na T90)")


if __name__ == "__main__":
    main()
