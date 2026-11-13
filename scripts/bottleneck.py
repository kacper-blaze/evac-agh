#!/usr/bin/env python3
"""Test wąskiego gardła: przepływ przez drzwi o szerokości b w funkcji ustawień modelu.

Pokój 12 x 12 m, drzwi w środku prawej ściany (z futryną), N osób rusza jednocześnie.
Mierzymy przepływ ustalony J (osoby między 20. a 80. percentylem czasów wyjścia) i przepływ
jednostkowy J/b [os/(m*s)]. Odniesienie z literatury (eksperymenty laboratoryjne, m.in. Jülich):
    Seyfried i in. (2009), "New insights into pedestrian flow through bottlenecks": J ~ 1,9 * b [1/s],
    typowy zakres dla drzwi ok. 1,3-1,9 os/(m*s) (Kretz i in. 2006; Hoogendoorn i Daamen 2005).
(źródła podane z pamięci - sprawdzić przed cytowaniem w raporcie).

  python scripts/bottleneck.py                 # tabela: ustawienia stare vs etap 2
  python scripts/bottleneck.py --widths 1.0 5.4 --n 300
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from evac.config import load_config  # noqa: E402
from evac.sim import Simulation  # noqa: E402

REF = (1.3, 1.9)

CASES = {
    # nazwa: nadpisania konfiguracji
    "stare, spokojnie (v0=1.3)": {"behaviour": {"panic_initial_share": 0.0}},
    "stare, panika (v_max=3.0)": {"behaviour": {"panic_initial_share": 1.0, "v_max_mps": 3.0}},
    "etap2, spokojnie": {"behaviour": {"panic_initial_share": 0.0, "density_speed": {"enabled": True}},
                         "map_options": {"gate_frame_depth_m": 0.8}},
    "etap2, panika (v_max=1.9)": {"behaviour": {"panic_initial_share": 1.0, "v_max_mps": 1.9,
                                                "density_speed": {"enabled": True}},
                                  "map_options": {"gate_frame_depth_m": 0.8}},
}


def room(b, size=12.0):
    y0, y1 = size / 2 - b / 2, size / 2 + b / 2
    return {"nazwa": f"wąskie gardło b={b}", "granica": [[0, 0], [size, 0], [size, size], [0, size]],
            "wyjscia": [{"id": "D", "szerokosc_m": b, "odcinek": [[size, y0], [size, y1]]}]}


def flow(b, over, n, seed, backend):
    base = {"seed": seed, "agents": {"count": n, "party_phase_s": 0, "group_share": 0.0},
            "behaviour": {"react_delay_s": [0, 0], "ignore_alarm_share": 0.0},
            "hazard": {"enabled": False}, "injury": {"enabled": False}, "sim": {"t_max_s": 400},
            "exits": {"hysteresis": 0.85}}
    from evac.config import _merge
    cfg, _ = load_config(None, _merge(base, over), [])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(room(b), f)
    try:
        res = Simulation(cfg, f.name, backend=backend, record=False).run()
    finally:
        os.unlink(f.name)
    te = np.sort(res.agents.t_exit[~np.isnan(res.agents.t_exit)])
    if len(te) < 0.9 * n:
        return float("nan"), len(te)
    i0, i1 = int(0.2 * len(te)), int(0.8 * len(te))
    return (i1 - i0) / max(te[i1] - te[i0], 1e-9), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 1.5, 2.5, 4.0, 5.4])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    print(f"Przepływ jednostkowy J/b [os/(m*s)]; literatura: {REF[0]}-{REF[1]} (Seyfried 2009: ~1.9)\n")
    print(f"{'ustawienia':30s}" + "".join(f"  b={b:<4}" for b in a.widths))
    for name, over in CASES.items():
        line = f"{name:30s}"
        for b in a.widths:
            js = [flow(b, over, a.n, s, a.backend)[0] / b for s in a.seeds]
            j = float(np.nanmean(js))
            rows.append({"ustawienia": name, "b_m": b, "J_na_metr": round(j, 2),
                         "w_zakresie_literatury": int(REF[0] * 0.8 <= j <= REF[1] * 1.2)})
            line += f"  {j:6.2f}"
        print(line, flush=True)
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Zapisano", a.csv)


if __name__ == "__main__":
    main()
