#!/usr/bin/env python3
"""Tworzy scenarios/mapa_dodatki.json - elementy, których nie ma w mapie zespołu (mapa_ms.json):

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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from evac.config import load_config  # noqa: E402
from evac.geometry import Scene  # noqa: E402

N_GRILLS, N_TREES, SEED = 6, 30, 2026


def main():
    new = json.load(open(os.path.join(ROOT, "scenarios", "mapa_ms.json"), encoding="utf-8"))
    old = json.load(open(os.path.join(ROOT, "scenarios", "mapa_ms_poprawiona.json"), encoding="utf-8"))
    border = Polygon(new["granica"])

    tunnels = []
    for t in old.get("tunele", []):
        q = np.asarray(t["obrys"], dtype=float)          # [wewn.1, wewn.2, zewn.2, zewn.1]
        vec = 0.5 * (q[2] + q[3]) - 0.5 * (q[0] + q[1])
        tunnels.append({"id": t["id"], "wyjscie": t["wyjscie"], "opis": t.get("opis", ""),
                        "zrodlo": t.get("zrodlo", "") + " (wektor ze starej mapy)", "aktywny": True,
                        "wektor": [round(float(vec[0]), 2), round(float(vec[1]), 2)]})

    benches = [b for b in old.get("lawki", []) if border.contains(Point(b["xy"]).buffer(0.5))]

    # --- przykładowe grille i drzewa: trawnik (poza chodnikami), część aktywna, z dala od ścian
    cfg, _ = load_config(None, {"map_options": {"gate_frame_depth_m": 0.8}}, [])
    data = dict(new, tunele=tunnels, lawki=benches)
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
    grills = spread(lawn(5.0), N_GRILLS, [], 25.0)
    trees = spread(lawn(2.0), N_TREES, bench_xy + [list(g) for g in grills], 4.0)
    note = "ROZMIESZCZENIE PRZYKŁADOWE - podmienić na współrzędne od zespołu"
    out = {
        "opis": "Dodatki do mapy zespołu (mapa_ms.json), dokładane przy wczytaniu. Generowane przez scripts/make_extras.py.",
        "tunele": tunnels,
        "lawki": benches,
        "grille": [{"id": f"G{k + 1}", "xy": [round(float(p[0]), 1), round(float(p[1]), 1)], "promien_m": 0.5,
                    "przykladowe": True, "uwaga": note} for k, p in enumerate(grills)],
        "drzewa": [{"id": f"D{k + 1}", "xy": [round(float(p[0]), 1), round(float(p[1]), 1)], "promien_m": 0.3,
                    "przykladowe": True, "uwaga": note} for k, p in enumerate(trees)],
    }
    path = os.path.join(ROOT, "scenarios", "mapa_dodatki.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Zapisano {path}: tunele {len(tunnels)}, ławki {len(benches)}/{len(old.get('lawki', []))}, "
          f"grille {len(grills)} (przykładowe), drzewa {len(trees)} (przykładowe)")


if __name__ == "__main__":
    main()
