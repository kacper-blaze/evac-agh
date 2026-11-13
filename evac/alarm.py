"""Alarm etapowy (alarm.mode = staged) i gapie.

Stany świadomości: nieświadomy -> zauważył -> zdecydował -> ucieka.
Skąd agent dowiaduje się o zdarzeniu:
  * fala informacji od zdarzenia (promień rośnie z prędkością `info_speed_mps`),
  * uciekający sąsiedzi ("poczta pantoflowa"),
  * dym o wyczuwalnym stężeniu, strefa wykluczenia, huk,
  * komunikat globalny po `global_delay_s` (brak przy awarii nagłośnienia).
Po zauważeniu: zwykli decydują i ruszają z opóźnieniem, panikujący od razu, ignorujący czekają na większość
sąsiadów (jak dotąd), gapie idą na punkt obserwacji przy zdarzeniu.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .agents import (AW_DECIDED, AW_NOTICED, AW_UNAWARE, IDLE, MOVING, T_GAWKER, T_IGNORER, T_PANICKER)


def _span(v):
    if v is None:
        return None
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return (float(v[0]), float(v[-1]))


def start(sim, tau):
    A, al, ev = sim.A, sim.cfg.alarm, sim.events
    sim.alarm_done = True
    sim.phys.e_open[:] = sim.open
    o = ev.origin
    A.dist_event[:] = np.hypot(A.x - o[0], A.y - o[1])
    A.tmove[:] = np.inf
    gd = _span(al.global_delay_s)
    sim.t_global = None if (gd is None or ev.pa_failure) else tau + float(sim.rng2.uniform(*gd))
    sim.gawk_peak = 0
    if not bool(al.keep_wandering):
        A.mode[:] = 0
        A.v0eff[:] = 0.0
    sim.res.events.append((tau, "zdarzenie (początek alarmu etapowego)"))


def notice(sim, idx, tau):
    """Agenci `idx` właśnie dowiedzieli się o zdarzeniu."""
    A, al, bc, ev = sim.A, sim.cfg.alarm, sim.cfg.behaviour, sim.events
    idx = np.asarray(idx, dtype=int)
    idx = idx[(A.aw[idx] == AW_UNAWARE) & (A.state[idx] == IDLE)]
    n = len(idx)
    if n == 0:
        return
    rng = sim.rng2
    A.aw[idx] = AW_NOTICED
    A.t_notice[idx] = tau
    A.mode[idx] = 0
    A.v0eff[idx] = 0.0
    typ = A.typ[idx]
    dd = rng.uniform(*_span(al.decide_delay_s), n)
    pp = rng.uniform(*_span(al.prep_delay_s), n)
    t_dec = tau + dd
    tm = t_dec + pp
    pan = typ == T_PANICKER
    tm[pan] = tau + rng.uniform(*_span(bc.panic_delay_s), int(pan.sum()))
    t_dec[pan] = tau
    A.p[idx[pan]] = np.maximum(A.p[idx[pan]], sim.PANICKER_P)
    ign = typ == T_IGNORER
    tm[ign] = np.inf
    t_dec[ign] = np.nan
    # gapie: tylko ci, którzy zauważyli zdarzenie z bliska, i tylko przy zdarzeniach "do oglądania"
    gk = bc.gawkers
    d = np.hypot(A.x[idx] - ev.origin[0], A.y[idx] - ev.origin[1])
    gaw = (typ == T_GAWKER) & ev.watchable & (d <= float(gk.max_dist_m))
    if gaw.any():
        g = idx[gaw]
        tm[gaw] = np.inf
        t_dec[gaw] = np.nan
        A.gawking[g] = True
        lo, hi = _span(gk.watch_dist_m)
        R = rng.uniform(lo, hi, len(g))
        for z in ev.active_zones(tau) + [z for z in ev.zones if tau < z["t"]]:
            R = np.maximum(R, z["rmax"] * 0.5 + 4.0)
        dg = np.maximum(d[gaw], 1e-6)
        ang = np.arctan2(A.y[g] - ev.origin[1], A.x[g] - ev.origin[0]) + rng.uniform(-0.35, 0.35, len(g))
        tx = ev.origin[0] + R * np.cos(ang)
        ty = ev.origin[1] + R * np.sin(ang)
        sc, wf = sim.scene, sim.wf
        iy, ix = sc.cell_of(tx, ty)
        iy, ix = np.clip(iy, 0, sc.ny - 1), np.clip(ix, 0, sc.nx - 1)
        ok = (dg > R) & (wf.walk[iy, ix] > 0) & (wf.d[iy, ix] >= 0.5) & sc.inside[iy, ix]
        if sim.reach is not None:
            ok &= sim.reach[iy, ix]
        go = g[ok]
        A.wx[go], A.wy[go] = tx[ok], ty[ok]
        A.mode[go] = 1
        A.v0eff[go] = 0.8 * A.v0[go]
    A.tmove[idx] = tm
    A.t_decide[idx] = t_dec


def step(sim, tau, dt):
    A, al, ev = sim.A, sim.cfg.alarm, sim.events
    rng = sim.rng2
    un = np.flatnonzero((A.aw == AW_UNAWARE) & (A.state == IDLE))
    glob = sim.t_global is not None and tau >= sim.t_global
    if glob:
        sim.res.events.append((tau, f"komunikat globalny (nieświadomych wcześniej: {len(un)})"))
        sim.t_global = None
    if len(un):
        m = np.zeros(len(un), dtype=bool)
        if tau >= ev.t0:                                    # fala informacji
            R = float(al.info_speed_mps) * (tau - ev.t0)
            wave = A.dist_event[un] <= R
            if al.wave_range_m is not None:
                wave &= A.dist_event[un] <= float(al.wave_range_m)
            m |= wave
        if glob:                                            # komunikat globalny
            m[:] = True
        wom = al.word_of_mouth
        mv = np.flatnonzero(A.state == MOVING)
        if len(mv) and float(wom.rate_per_s) > 0 and not m.all():
            tree = cKDTree(np.column_stack((A.x[mv], A.y[mv])))
            k = tree.query_ball_point(np.column_stack((A.x[un], A.y[un])), r=float(wom.radius_m),
                                      return_length=True)
            m |= rng.random(len(un)) < 1.0 - np.exp(-float(wom.rate_per_s) * k * dt)
        m |= A.smoke[un] > float(al.smoke_notice_conc)
        for z in ev.active_zones(tau):
            m |= np.hypot(A.x[un] - z["xy"][0], A.y[un] - z["xy"][1]) <= z["R"] + 3.0
        notice(sim, un[m], tau)
    A.aw[(A.aw == AW_NOTICED) & (A.t_decide <= tau)] = AW_DECIDED
    # ---- gapie
    A.gawking &= A.state == IDLE
    g = np.flatnonzero(A.gawking)
    sim.gawk_peak = max(sim.gawk_peak, len(g))
    if len(g):
        gk = sim.cfg.behaviour.gawkers
        watch = tau - A.t_notice[g]
        leave = (A.smoke[g] > float(gk.leave_smoke_conc)) | (A.p[g] > float(gk.leave_panic)) \
            | (watch > float(gk.max_watch_s))
        if len(g) < float(gk.leave_frac_of_peak) * sim.gawk_peak:
            leave |= watch > float(gk.min_watch_s)
        for z in ev.active_zones(tau):
            leave |= np.hypot(A.x[g] - z["xy"][0], A.y[g] - z["xy"][1]) <= z["R"] + 2.0
        tired = g[(A.mode[g] == 1) & (watch > 40.0)]        # nie doszedł na punkt (przeszkoda) - ogląda stąd
        A.mode[tired] = 0
        A.v0eff[tired] = 0.0
        L = g[leave]
        A.gawking[L] = False
        A.mode[L] = 0
        A.v0eff[L] = 0.0
        A.tmove[L] = tau + rng.uniform(0.0, 2.0, len(L))
