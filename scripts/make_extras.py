#!/usr/bin/env python3
"""Tworzy scenarios/mapa_dodatki.json - elementy, których nie ma w mapie scenariusza etap2.yaml:

  * tunele  - przejścia przez budynki za W1-W3: wektor wyciągnięcia odcinka bramy, przeniesiony ze starej mapy
              (dzięki temu korytarz dopasowuje się do bramy, gdy zespół zmieni jej położenie lub szerokość),
  * lawki   - ławki z OpenStreetMap przeniesione ze starej mapy (tylko te wewnątrz nowego obrysu),
  * grille, drzewa - rozmieszczenie PRZYKŁADOWE ("przykladowe": true) do podmiany na prawdziwe współrzędne.

Mapy zespołu nie zmieniamy: dodatki są dokładane przy wczytaniu (`map_extras:` w YAML).
Jeśli mapa zespołu sama zawiera sekcję "tunele" / "lawki" / "grille" / "drzewa", wygrywa mapa.

  python scripts/make_extras.py            # wymaga shapely (jak mapa_ms.py)
"""
import json
import os
import sys

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from evac.config import load_config  # noqa: E402
from evac.geometry import Scene  # noqa: E402

GRILL_TARGETS = [
    (94.0, 186.0), (11.0, 66.0), (78.0, 236.0), (26.0, 153.0),
    (70.0, 107.0), (52.0, 184.0), (67.0, 160.0), (92.0, 220.0),
    (56.0, 212.0), (48.0, 212.0), (42.0, 124.0), (89.0, 117.0),
]
N_TREES, SEED = 30, 2026


def main():
    scenario_path = os.path.join(ROOT, "scenarios", "etap2.yaml")
    cfg, _ = load_config(scenario_path, {"map_options": {"gate_frame_depth_m": 0.8}}, [])
    map_path = cfg.map if os.path.isabs(cfg.map) else os.path.join(ROOT, "scenarios", cfg.map)
    with open(map_path, encoding="utf-8") as f:
        new = json.load(f)
    extras_path = os.path.join(ROOT, "scenarios", "mapa_dodatki.json")
    if os.path.isfile(extras_path):
        with open(extras_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    border = Polygon(new["granica"])
    tunnels = new.get("tunele", existing.get("tunele", []))
    benches = new.get("lawki", existing.get("lawki", []))
    bench_count = len(benches)
    benches = [b for b in benches if border.contains(Point(b["xy"]).buffer(0.5))]

    # --- przykładowe grille i drzewa: trawnik (poza chodnikami), część aktywna, z dala od ścian
    data = dict(new)
    data.setdefault("tunele", tunnels)
    data.setdefault("lawki", benches)
    sc = Scene(data, cfg)
    wf = sc.wall_fields(sc.initial_open())
    reach = sc.reachable(wf, sc.initial_open())
    walks = unary_union([LineString(c["os"]).buffer(c["szerokosc_m"] / 2 + 1.0) if "os" in c
                         else Polygon(c["obrys"]).buffer(1.0) for c in new.get("chodniki", [])])
    rng = np.random.default_rng(SEED)

    def lawn(min_wall, n_cand=4000):
        pts = sc.free_points(wf, min_wall, mask=reach)
        pts = pts[rng.choice(len(pts), min(n_cand, len(pts)), replace=False)]
        return np.array([p for p in pts if not walks.contains(Point(p))])

    def spread(cand, k, taken, min_sep):
        """Wybór k punktów metodą "najdalszy od już wybranych"."""
        chosen = list(taken)
        out = []
        for _ in range(k):
            if chosen:
                d = np.min(np.hypot(cand[:, None, 0] - np.array(chosen)[None, :, 0],
                                    cand[:, None, 1] - np.array(chosen)[None, :, 1]), axis=1)
            else:
                d = rng.random(len(cand)) + min_sep
            d = d * (0.6 + 0.4 * rng.random(len(cand)))          # trochę nieregularności
            i = int(np.argmax(d))
            out.append(cand[i]); chosen.append(cand[i])
        return out

    bench_xy = [b["xy"] for b in benches]
    free = sc.free_points(wf, 2.0, mask=reach)
    if not len(free):
        raise RuntimeError("Flagi mapy nie pozostawiają dostępnego terenu dla grilli i drzew.")
    free_tree = cKDTree(free)
    grill_targets = np.asarray(GRILL_TARGETS)
    grill_dist, grill_idx = free_tree.query(grill_targets)
    if np.any(grill_dist > 5.0):
        raise ValueError("Punkt grilla jest zbyt daleko od wolnego terenu mapy.")
    grills = free_tree.data[grill_idx]
    trees = spread(lawn(2.0), N_TREES, bench_xy + [list(g) for g in grills], 4.0)
    grill_note = "Położenie orientacyjne przy ścieżkach i wokół placu asfaltowego; niepotwierdzone pomiarem"
    tree_note = "ROZMIESZCZENIE PRZYKŁADOWE - podmienić na współrzędne od zespołu"
    out = {
        "opis": f"Dodatki do mapy zespołu ({os.path.basename(map_path)}), dokładane przy wczytaniu. "
                "Generowane przez scripts/make_extras.py.",
        "tunele": tunnels,
        "lawki": benches,
        "grille": [{"id": f"G{k + 1}", "xy": [round(float(p[0]), 1), round(float(p[1]), 1)], "promien_m": 0.5,
                    "przykladowe": True, "uwaga": grill_note} for k, p in enumerate(grills)],
        "drzewa": [{"id": f"D{k + 1}", "xy": [round(float(p[0]), 1), round(float(p[1]), 1)], "promien_m": 0.3,
                    "przykladowe": True, "uwaga": tree_note} for k, p in enumerate(trees)],
    }
    path = extras_path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Zapisano {path}: tunele {len(tunnels)}, ławki {len(benches)}/{bench_count}, "
          f"grille {len(grills)} (przykładowe), drzewa {len(trees)} (przykładowe)")


if __name__ == "__main__":
    main()
