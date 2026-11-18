#!/usr/bin/env python3
"""Seria przebiegów: warianty mapy x liczba osób x ziarna (x zdarzenia) -> tabele porównawcze.

  python run_matrix.py scenarios/etap2.yaml --n 3000 6000 10000 --workers 4
  python run_matrix.py scenarios/etap2.yaml --n 3000 --variants normalny remont --events pozar huk --seeds 1 2 3
  python run_matrix.py scenarios/juwenalia.yaml --gates wszystkie_otwarte Z1_zamknieta      # stary tryb (stara mapa)

Wyniki w results/:
  matrix.csv              każdy przebieg osobno
  matrix_summary.csv/.md  średnia ± połowa 95% przedziału ufności (t-Studenta) po ziarnach
Liczba ziaren domyślnie zależy od liczby osób (10 dla <= 3000, 5 dla <= 6000, 3 powyżej) - ograniczenie czasu
obliczeń; własną listę podaje --seeds. Każde ziarno to inne rozmieszczenie ludzi, inne miejsce zdarzenia
(`at: grill` / `at: random`) i inny kierunek wiatru.
"""
import argparse, csv, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATE_SCENARIOS = {"wszystkie_otwarte": [], "Z1_zamknieta": ["Z1"], "Z2_zamknieta": ["Z2"], "Z1_i_Z2_zamkniete": ["Z1", "Z2"]}
VARIANTS = ["normalny", "remont", "czwartek", "remont_czwartek"]
METRICS = ["odsetek_ewakuowanych_%", "T50_s", "T90_s", "czas_ewakuacji_zdolnych_s", "wyszli_cali", "wyszli_ranni",
           "nie_wyszli_ciezko_ranni", "nie_wyszli_pozostali", "przewroceni_choc_raz", "max_cisnienie_Npm",
           "max_szczyt_przeplywu_os_m_s", "gapie_szczyt", "zmienili_brame_%", "korelacja_ruszenie_odleglosc"]


def auto_seeds(n, base=1):
    k = 10 if n <= 3000 else 5 if n <= 6000 else 3
    return list(range(base, base + k))


def one(job):
    from evac.render import render_video
    from evac.sim import Simulation
    from evac.stats import write_outputs
    from run import build_cfg, make_tag, resolve_map
    scen, n, name, closed, variant, event, seed, outdir, backend, video, sets = job
    a = argparse.Namespace(scenario=scen, n=n, seed=seed, closed=closed, tmax=None, party=None, set=sets,
                           variant=variant, event=event)
    cfg, _ = build_cfg(a)
    tag = make_tag(cfg) if variant else f"N{n}_{name}_s{seed}"
    sim = Simulation(cfg, resolve_map(cfg, scen), backend=backend, record=video)
    res = sim.run()
    out = write_outputs(res, outdir, tag)
    if video:
        render_video(res, os.path.join(outdir, tag + ".mp4"), fps=int(cfg.output.fps))
    row = {"scenariusz_bram": name, "zestaw_zdarzen": event or "-", **out["summary"]}
    print(f"   gotowe: {tag}  ({res.wall_time_s:.0f} s)", flush=True)
    return row


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def aggregate(rows):
    """Średnia ± połowa 95% CI po ziarnach dla każdej kombinacji (wariant, zdarzenia, N)."""
    from scipy.stats import t as student
    groups = {}
    for r in rows:
        groups.setdefault((r["scenariusz_bram"], r["zestaw_zdarzen"], r["N"]), []).append(r)
    out = []
    for (name, event, n), rs in groups.items():
        row = {"wariant": name, "zdarzenia": event, "N": n, "ziaren": len(rs)}
        for m in METRICS:
            v = np.array([_num(r.get(m)) for r in rs])
            v = v[~np.isnan(v)]
            if len(v) == 0:
                row[m] = ""
                continue
            ci = student.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            row[m] = round(float(v.mean()), 2)
            row[m + "_ci95"] = round(float(ci), 2) if len(v) > 1 else ""
        out.append(row)
    return out


def write_md(path, agg):
    cols = [("odsetek_ewakuowanych_%", "ewakuowani %"), ("T50_s", "T50 [s]"), ("T90_s", "T90 [s]"),
            ("czas_ewakuacji_zdolnych_s", "T100 [s]"), ("wyszli_ranni", "wyszli ranni"),
            ("nie_wyszli_ciezko_ranni", "nie wyszli (ciężko ranni)"), ("nie_wyszli_pozostali", "zostali (sprawni)"),
            ("max_szczyt_przeplywu_os_m_s", "szczyt przepływu [os/(m·s)]")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| wariant | zdarzenia | N | ziaren | " + " | ".join(c[1] for c in cols) + " |\n")
        f.write("|" + "---|" * (4 + len(cols)) + "\n")
        for r in agg:
            cells = []
            for key, _ in cols:
                ci = r.get(key + "_ci95", "")
                cells.append(f"{r[key]}" + (f" ± {ci}" if ci != "" else ""))
            f.write(f"| {r['wariant']} | {r['zdarzenia']} | {r['N']} | {r['ziaren']} | " + " | ".join(cells) + " |\n")
        f.write("\nŚrednia ± połowa 95% przedziału ufności po ziarnach. Parametry modelu są ilustracyjne (niekalibrowane).\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario")
    ap.add_argument("--n", type=int, nargs="+", default=[3000, 6000, 10000])
    ap.add_argument("--variants", nargs="+", default=None, help=f"warianty mapy (domyślnie: {' '.join(VARIANTS)})")
    ap.add_argument("--gates", nargs="+", default=None, choices=list(GATE_SCENARIOS),
                    help="stary tryb: konfiguracje bram zamiast wariantów")
    ap.add_argument("--events", nargs="+", default=[None], help="zestawy zdarzeń z event_presets (np. pozar huk pakunek)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="lista ziaren (domyślnie zależna od N)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--video", action="store_true", help="wideo dla każdego przebiegu (wolne)")
    ap.add_argument("--video-first", action="store_true", help="wideo tylko dla pierwszego ziarna każdej kombinacji")
    ap.add_argument("--set", action="append", default=[], help="nadpisanie: klucz.podklucz=wartosc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.gates:
        combos = [(g, GATE_SCENARIOS[g], None) for g in a.gates]
    else:
        combos = [(v, None, v) for v in (a.variants or VARIANTS)]
    jobs = []
    for n in a.n:
        seeds = a.seeds or auto_seeds(n)
        for name, closed, variant in combos:
            for event in a.events:
                for i, s in enumerate(seeds):
                    video = a.video or (a.video_first and i == 0)
                    jobs.append((a.scenario, n, name, closed, variant, event, s, a.out, a.backend, video, a.set))
    print(f"Przebiegów: {len(jobs)}", flush=True)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, jobs))
    else:
        rows = [one(j) for j in jobs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = os.path.join(a.out, "matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    agg = aggregate(rows)
    keys = []
    for r in agg:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(a.out, "matrix_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(agg)
    write_md(os.path.join(a.out, "matrix_summary.md"), agg)
    print("Zapisano", path, "oraz matrix_summary.csv / .md")


if __name__ == "__main__":
    main()
