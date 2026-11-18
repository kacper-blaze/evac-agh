"""Zdarzenia: ogólna lista `events:` z YAML-a (typ, czas, pozycja, parametry).

Typy:
  fire        pożar: źródło dymu (parametry dymu w sekcji `hazard`)
  bang        huk / petarda: fala paniki bez dymu
  package     podejrzany pakunek: strefa wykluczenia rosnąca z czasem
  gate_block  blokada bramy w trakcie (odpowiednik exits.close_at_s)
  pa_failure  awaria nagłośnienia: brak komunikatu globalnego (informacja tylko "pocztą pantoflową")
  storm       nagła burza: wszyscy od razu szukają schronienia - śpieszą się (`hurry`) i wolą zadaszone
              przejścia przez budynki (`shelter_bonus_m`); bez dymu i bez gapiów

Pozycja: `xy: [x, y]` albo `at: grill` (losowy grill z mapy) / `at: random` (losowy punkt części aktywnej,
opcjonalnie `region: [xmin, xmax, ymin, ymax]`) / `at: [[x, y], [x, y], ...]` (losowo z listy miejsc).
Losowanie zależy od ziarna (`seed`), więc seria ziaren daje różne miejsca zdarzenia.

Pusta lista = dotychczasowe zachowanie: pożar w `hazard.origin_xy` (o ile `hazard.enabled`).
"""
from __future__ import annotations

import warnings

import numpy as np

WATCHABLE = ("fire", "package")      # zdarzenia, które przyciągają gapiów
LABELS = {"fire": "pożar", "bang": "huk", "package": "pakunek"}


class Events:
    def __init__(self, sim):
        self.sim = sim
        cfg, sc = sim.cfg, sim.scene
        self.rng = np.random.default_rng([int(cfg.seed), 101])
        self.fires, self.bangs, self.zones, self.all, self.storms = [], [], [], [], []
        self.pa_failure = False
        raw = [dict(e) for e in (cfg.events or [])]
        self.legacy = not raw
        if self.legacy:
            if cfg.hazard.enabled:
                self._add({"type": "fire", "t": float(cfg.hazard.start_s),
                           "xy": np.asarray(cfg.hazard.origin_xy, dtype=float), "label": str(cfg.hazard.source_label)})
        for ev in raw:
            typ = str(ev.get("type", "fire"))
            t = float(ev.get("t_s", 0.0))
            if typ == "pa_failure":
                self.pa_failure = True
                sim.pending_log.append((t, "awaria nagłośnienia (brak komunikatu globalnego)"))
            elif typ == "gate_block":
                for g in np.atleast_1d(ev.get("gate", [])):
                    sim.close_events.append((t, str(g)))
                sim.close_events.sort()
            elif typ == "storm":
                self.storms.append({"type": "storm", "t": t, "xy": None, "label": str(ev.get("label", "burza")),
                                    "done": False, "hurry": float(ev.get("hurry", 1.25)),
                                    "shelter_bonus_m": float(ev.get("shelter_bonus_m", 30.0)),
                                    "panic": float(ev.get("panic", 0.15))})
            elif typ in ("fire", "bang", "package"):
                item = {"type": typ, "t": t, "xy": self._pos(ev), "label": str(ev.get("label", LABELS[typ])),
                        "done": False}
                if typ == "bang":
                    item.update(panic_peak=float(ev.get("panic_peak", 0.7)), scale_m=float(ev.get("scale_m", 40.0)),
                                hear_radius_m=ev.get("hear_radius_m"))
                if typ == "package":
                    item.update(r0=float(ev.get("r0_m", 8.0)), growth=float(ev.get("growth_mps", 0.25)),
                                rmax=float(ev.get("r_max_m", 40.0)), close_gates=bool(ev.get("close_gates", True)),
                                cost=float(ev.get("zone_cost", 60.0)), push=float(ev.get("push_mps2", 2.5)), R=0.0)
                self._add(item)
            else:
                warnings.warn(f"Nieznany typ zdarzenia '{typ}' - pomijam.")
        main = [e for e in self.all] + self.storms
        self.primary = min(main, key=lambda e: e["t"]) if main else None
        if self.primary is not None and self.primary["xy"] is None:      # burza nie ma miejsca
            self.origin, self.t0 = sim.free_pts.mean(axis=0), self.primary["t"]
        elif self.primary is not None:
            self.origin, self.t0 = self.primary["xy"], self.primary["t"]
        else:
            self.origin, self.t0 = np.asarray(cfg.hazard.origin_xy, dtype=float), 0.0
        self.watchable = self.primary is not None and self.primary["type"] in WATCHABLE

    def _add(self, item):
        self.all.append(item)
        {"fire": self.fires, "bang": self.bangs, "package": self.zones}[item["type"]].append(item)

    def _pos(self, ev) -> np.ndarray:
        sim = self.sim
        if "xy" in ev:
            return np.asarray(ev["xy"], dtype=float)
        at = ev.get("at", "random")
        pts = sim.free_pts
        if isinstance(at, (list, tuple)):
            cand = np.asarray(at, dtype=float).reshape(-1, 2)
            return cand[self.rng.integers(len(cand))]
        if at == "grill":
            g = np.array([q["xy"] for q in sim.scene.grills], dtype=float).reshape(-1, 2)
            if len(g) and len(pts):       # tylko grille w części aktywnej
                d = np.min(np.hypot(g[:, None, 0] - pts[None, ::7, 0], g[:, None, 1] - pts[None, ::7, 1]), axis=1)
                g = g[d < 3.0]
            if len(g):
                return g[self.rng.integers(len(g))]
            warnings.warn("Zdarzenie 'at: grill', ale mapa nie ma grilli w części aktywnej - losuję punkt.")
        if "region" in ev:
            x0, x1, y0, y1 = map(float, ev["region"])
            m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            if m.any():
                pts = pts[m]
        return pts[self.rng.integers(len(pts))].copy()

    # ------------------------------------------------------------------ krok
    def step(self, tau: float, dt: float):
        sim, A = self.sim, self.sim.A
        from .agents import EVACUATED, IDLE
        for b in self.bangs:
            if b["done"] or tau < b["t"]:
                continue
            b["done"] = True
            alive = np.flatnonzero(A.state != EVACUATED)
            d = np.hypot(A.x[alive] - b["xy"][0], A.y[alive] - b["xy"][1])
            dp = b["panic_peak"] * np.exp(-d / b["scale_m"]) / A.rho[alive]
            A.p[alive] = np.clip(A.p[alive] + dp * (1 - A.p[alive]), 0.0, 1.0)
            if sim.staged:
                hear = alive if b["hear_radius_m"] is None else alive[d <= float(b["hear_radius_m"])]
                sim.notice(hear[A.state[hear] == IDLE], tau)
            sim.res.events.append((tau, f"{b['label']} w ({b['xy'][0]:.0f}, {b['xy'][1]:.0f})"))
        for z in self.zones:
            if tau < z["t"]:
                continue
            if not z["done"]:
                z["done"] = True
                sim.res.events.append((tau, f"{z['label']} w ({z['xy'][0]:.0f}, {z['xy'][1]:.0f}) - strefa wykluczenia"))
            z["R"] = min(z["r0"] + z["growth"] * (tau - z["t"]), z["rmax"])
            if z["close_gates"]:
                for k, e in enumerate(sim.scene.exits):
                    if sim.open[k] == 1 and sim.open.sum() > 1 and np.hypot(*(e.center - z["xy"])) <= z["R"]:
                        sim.close_gate(k, tau, f"{e.id} w strefie wykluczenia - zamknięta")
        for s in self.storms:
            if s["done"] or tau < s["t"]:
                continue
            s["done"] = True
            sim.hurry = s["hurry"]
            for k, e in enumerate(sim.scene.exits):
                if e.id in sim.scene.covered_exits:
                    sim.gate_bonus[k] = s["shelter_bonus_m"]
            alive = np.flatnonzero(A.state != EVACUATED)
            A.p[alive] = np.clip(A.p[alive] + s["panic"] / A.rho[alive] * (1 - A.p[alive]), 0.0, 1.0)
            if sim.staged:
                sim.notice(np.flatnonzero(A.state == IDLE), tau)
            sim.res.events.append((tau, f"{s['label']} - wszyscy szukają schronienia"))
        for f in self.fires:
            if not f.get("done") and tau >= f["t"]:
                f["done"] = True
                if not self.legacy:
                    sim.res.events.append((tau, f"{f['label']} w ({f['xy'][0]:.0f}, {f['xy'][1]:.0f})"))

    def active_zones(self, tau):
        return [z for z in self.zones if tau >= z["t"]]
