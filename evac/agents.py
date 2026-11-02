"""Tworzenie agentów: pozycje (losowo po całym obrysie), grupy, typy zachowania, cechy indywidualne."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# stany agenta
IDLE, MOVING, EVACUATED, FALLEN = 0, 1, 2, 3
# typy zachowania
T_NORMAL, T_IGNORER, T_PANICKER = 0, 1, 2


@dataclass
class Agents:
    n: int
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    r: np.ndarray
    m: np.ndarray
    v0: np.ndarray          # prędkość ucieczki (bazowa)
    vw: np.ndarray          # prędkość spaceru na imprezie
    aggr: np.ndarray        # agresja 0..1
    rho: np.ndarray         # odporność na stres (większa = mniej podatny)
    p: np.ndarray           # poziom paniki 0..1
    typ: np.ndarray         # int8
    state: np.ndarray       # int8
    mode: np.ndarray        # int8: 0 = stoi, 1 = idzie do punktu (na imprezie)
    gid: np.ndarray         # int32: numer grupy
    tmove: np.ndarray       # planowany czas ruszenia (inf dla ignorujących)
    target: np.ndarray      # int32: wybrana brama
    v0eff: np.ndarray       # aktualna prędkość pożądana
    wx: np.ndarray          # cel spaceru
    wy: np.ndarray
    cx: np.ndarray          # środek grupy (siła spójności)
    cy: np.ndarray
    coh: np.ndarray         # siła spójności (0 = brak)
    press_t: np.ndarray     # jak długo ciśnienie przekracza próg
    pressure: np.ndarray    # ostatnie ciśnienie [N/m]
    exit_k: np.ndarray      # int32: którą bramą wyszedł (-1 = jeszcze nie)
    t_exit: np.ndarray      # czas wyjścia
    t_fall: np.ndarray      # czas przewrócenia
    t_start: np.ndarray     # czas ruszenia
    smoke: np.ndarray       # ostatnie stężenie dymu w punkcie agenta
    n_groups: int = 0


def _truncnorm(rng, mean, sd, lo, hi, n):
    return np.clip(rng.normal(mean, sd, n), lo, hi)


def create_agents(scene, wf, cfg, rng: np.random.Generator) -> Agents:
    ac = cfg.agents
    N = int(ac.count)
    r = rng.uniform(ac.radius_m[0], ac.radius_m[1], N)
    free = scene.free_points(wf, float(ac.min_wall_dist_m), rng=rng)
    if len(free) == 0:
        raise RuntimeError("Brak wolnych punktów w obrysie - sprawdź mapę.")

    # --- grupy ---
    gid = -np.ones(N, dtype=np.int32)
    order = rng.permutation(N)
    n_grouped = int(round(ac.group_share * N))
    pos = 0
    g = 0
    lo, hi = int(ac.group_size[0]), int(ac.group_size[1])
    while pos < n_grouped:
        size = int(rng.integers(lo, hi + 1))
        size = min(size, n_grouped - pos)
        if size < 2:
            break
        gid[order[pos: pos + size]] = g
        pos += size
        g += 1
    singles = np.flatnonzero(gid < 0)
    gid[singles] = g + np.arange(len(singles))
    n_groups = g + len(singles)

    centers = free[rng.integers(0, len(free), n_groups)]
    x = centers[gid, 0].copy()
    y = centers[gid, 1].copy()
    grouped = np.zeros(N, dtype=bool)
    sizes = np.bincount(gid, minlength=n_groups)
    grouped = sizes[gid] >= 2
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = rng.uniform(0.3, 0.9, N) * np.sqrt(sizes[gid])
    x = np.where(grouped, x + rad * np.cos(ang), x)
    y = np.where(grouped, y + rad * np.sin(ang), y)
    x += rng.normal(0, 0.02, N)
    y += rng.normal(0, 0.02, N)

    # --- rozsunięcie nakładających się agentów i pilnowanie ścian ---
    cw = scene.cw

    def ok_cells(px, py):
        iy, ix = scene.cell_of(px, py)
        good = (iy >= 0) & (iy < scene.ny) & (ix >= 0) & (ix < scene.nx)
        iyc = np.clip(iy, 0, scene.ny - 1)
        ixc = np.clip(ix, 0, scene.nx - 1)
        return good & (wf.walk[iyc, ixc] > 0) & scene.inside[iyc, ixc] & (wf.d[iyc, ixc] >= r + 0.1)

    bad = ~ok_cells(x, y)
    if bad.any():
        idx = np.flatnonzero(bad)
        pts = free[rng.integers(0, len(free), len(idx))]
        x[idx], y[idx] = pts[:, 0], pts[:, 1]
    for _ in range(80):
        tree = cKDTree(np.column_stack((x, y)))
        pairs = tree.query_pairs(r=0.8, output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.hypot(dx, dy)
        overlap = (r[i] + r[j] + 0.03) - d
        m = overlap > 0
        if not m.any():
            break
        i, j, dx, dy, d, overlap = i[m], j[m], dx[m], dy[m], d[m], overlap[m]
        d = np.maximum(d, 1e-6)
        nxv, nyv = dx / d, dy / d
        # przy d~0 losowy kierunek
        deg = d < 1e-4
        if deg.any():
            a = rng.uniform(0, 2 * np.pi, deg.sum())
            nxv[deg], nyv[deg] = np.cos(a), np.sin(a)
        shift_x = np.zeros(N)
        shift_y = np.zeros(N)
        np.add.at(shift_x, i, 0.5 * overlap * nxv)
        np.add.at(shift_y, i, 0.5 * overlap * nyv)
        np.add.at(shift_x, j, -0.5 * overlap * nxv)
        np.add.at(shift_y, j, -0.5 * overlap * nyv)
        nx_, ny_ = x + shift_x, y + shift_y
        good = ok_cells(nx_, ny_)
        x = np.where(good, nx_, x)
        y = np.where(good, ny_, y)

    # --- typy zachowania ---
    bc = cfg.behaviour
    typ = np.zeros(N, dtype=np.int8)
    perm = rng.permutation(N)
    n_pan = int(round(bc.panic_initial_share * N))
    n_ign = int(round(bc.ignore_alarm_share * N))
    typ[perm[:n_pan]] = T_PANICKER
    typ[perm[n_pan: n_pan + n_ign]] = T_IGNORER

    v0 = _truncnorm(rng, ac.v0_mps.mean, ac.v0_mps.sd, ac.v0_mps.min, ac.v0_mps.max, N)
    vw = _truncnorm(rng, ac.wander_speed_mps, 0.15, 0.2, 1.0, N)
    aggr = np.clip(rng.normal(ac.aggression.mean, ac.aggression.sd, N), 0.0, 1.0)
    rho = np.clip(rng.normal(ac.stress_resistance.mean, ac.stress_resistance.sd, N), 0.4, 2.0)

    z = np.zeros(N)
    A = Agents(
        n=N, x=x, y=y, vx=z.copy(), vy=z.copy(), r=r, m=np.full(N, float(cfg.sfm.mass_kg)),
        v0=v0, vw=vw, aggr=aggr, rho=rho, p=z.copy(), typ=typ,
        state=np.zeros(N, dtype=np.int8), mode=np.zeros(N, dtype=np.int8), gid=gid,
        tmove=np.full(N, np.inf), target=-np.ones(N, dtype=np.int32), v0eff=z.copy(),
        wx=x.copy(), wy=y.copy(), cx=x.copy(), cy=y.copy(), coh=z.copy(),
        press_t=z.copy(), pressure=z.copy(), exit_k=-np.ones(N, dtype=np.int32),
        t_exit=np.full(N, np.nan), t_fall=np.full(N, np.nan), t_start=np.full(N, np.nan),
        smoke=z.copy(), n_groups=n_groups,
    )
    return A
