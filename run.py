#!/usr/bin/env python3
"""Uruchomienie pojedynczej symulacji.

Przykłady:
  python run.py scenarios/juwenalia.yaml
  python run.py scenarios/juwenalia.yaml --n 6000 --closed Z2 --seed 7
  python run.py scenarios/juwenalia.yaml --set hazard.origin_xy=[100,200] --set behaviour.ignore_alarm_share=0.3
  python run.py scenarios/etap2.yaml --variant remont_czwartek --event huk     # nowa mapa, moduły etapu 2
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evac.config import load_config  # noqa: E402
from evac.render import render_video  # noqa: E402
from evac.sim import Simulation  # noqa: E402
from evac.stats import write_outputs  # noqa: E402


def resolve_map(cfg, yaml_path):
    cand = [cfg.map]
    if yaml_path:
        cand.append(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), cfg.map))
    cand.append(os.path.join(HERE, "scenarios", cfg.map))
    for c in cand:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Nie znaleziono mapy: {cfg.map}")


def build_cfg(args):
    over = {}
    if args.n is not None:
        over.setdefault("agents", {})["count"] = args.n
    if args.seed is not None:
        over["seed"] = args.seed
    if args.closed:
        over.setdefault("exits", {})["closed"] = args.closed
    if args.tmax is not None:
        over.setdefault("sim", {})["t_max_s"] = args.tmax
    if args.party is not None:
        over.setdefault("agents", {})["party_phase_s"] = args.party
    return load_config(args.scenario, over, args.set, variant=getattr(args, "variant", None),
                       event=getattr(args, "event", None))


def make_tag(cfg):
    closed = "_".join(cfg.exits.closed) if cfg.exits.closed else "all"
    parts = [f"N{cfg.agents.count}", cfg.variant if cfg.variant else f"closed-{closed}"]
    if cfg.event_name:
        parts.append(str(cfg.event_name))
    return "_".join(parts + [f"s{cfg.seed}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", default=None, help="plik YAML (brak = ustawienia domyślne)")
    ap.add_argument("--n", type=int, help="liczba osób")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--closed", nargs="*", help="bramy zamknięte od początku, np. Z1 Z2")
    ap.add_argument("--variant", help="wariant mapy: normalny | remont | czwartek | remont_czwartek (lub własny z YAML)")
    ap.add_argument("--event", help="zestaw zdarzeń z event_presets w YAML, np. pozar | huk | pakunek")
    ap.add_argument("--tmax", type=float, help="maks. czas od alarmu [s]")
    ap.add_argument("--party", type=float, help="czas fazy imprezy przed alarmem [s]")
    ap.add_argument("--set", action="append", default=[], help="nadpisanie: klucz.podklucz=wartosc")
    ap.add_argument("--out", default=None, help="folder wyników")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--backend", choices=["auto", "numba", "numpy"], default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    cfg, _ = build_cfg(args)
    outdir = args.out or os.path.join(HERE, cfg.output.dir)
    tag = args.tag or make_tag(cfg)
    map_path = resolve_map(cfg, args.scenario)

    def progress(tau, ev, mv, fl, wall):
        print(f"  t={tau:7.1f}s  ewakuowani={ev:5d}  w ruchu={mv:5d}  przewróceni={fl:3d}  ({wall:.0f}s obliczeń)",
              flush=True)

    print(f"Mapa: {map_path}\nN={cfg.agents.count}  zamknięte: {cfg.exits.closed or '-'}  seed={cfg.seed}")
    sim = Simulation(cfg, map_path, backend=args.backend, progress=progress,
                     record=bool(cfg.output.video) and not args.no_video)
    print(f"Backend fizyki: {sim.phys.backend}")
    ar = sim.area
    print(f"Teren: aktywne {ar['aktywne_m2']:.0f} m2, wyłączone (odcięte) {ar['wylaczone_m2']:.0f} m2; "
          f"gęstość początkowa {cfg.agents.count / max(ar['aktywne_m2'], 1):.2f} os/m2")
    for e in sim.events.storms:
        print(f"Zdarzenie: {e['label']} t={e['t']:.0f}s (cały teren)")
    for e in sim.events.all:
        print(f"Zdarzenie: {e['label']} t={e['t']:.0f}s w ({e['xy'][0]:.1f}, {e['xy'][1]:.1f})")
    res = sim.run()
    out = write_outputs(res, outdir, tag)
    print("\n=== PODSUMOWANIE ===")
    for k, v in out["summary"].items():
        print(f"{k:38s} {v}")
    if res.frames:
        print("Renderuję wideo...", flush=True)
        vp = render_video(
            res, os.path.join(outdir, f"{tag}.mp4"), fps=int(cfg.output.fps),
            progress_every=int(cfg.output.render_progress_every),
        )
        print("Wideo:", vp)
    print("Wyniki w:", outdir)
    return res


if __name__ == "__main__":
    main()
