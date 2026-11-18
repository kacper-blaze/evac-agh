"""Tworzenie agentów: pozycje (losowo po całym obrysie), grupy, typy zachowania, cechy indywidualne."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# stany agenta
IDLE, MOVING, EVACUATED, FALLEN = 0, 1, 2, 3
# typy zachowania
T_NORMAL, T_IGNORER, T_PANICKER, T_GAWKER = 0, 1, 2, 3
# świadomość (alarm etapowy)
AW_UNAWARE, AW_NOTICED, AW_DECIDED, AW_FLEEING = 0, 1, 2, 3
# obrażenia (model stopniowany)
INJ_NONE, INJ_LIGHT, INJ_SEVERE = 0, 1, 2


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
    # --- pola modułów rozszerzonych (wypełniane w __post_init__ / create_agents)
    axe: np.ndarray = None      # przyspieszenie zewnętrzne (dym, strefa wykluczenia)
    aye: np.ndarray = None
    aw: np.ndarray = None       # int8: świadomość zdarzenia (AW_*)
    t_notice: np.ndarray = None # kiedy zauważył zdarzenie
    t_decide: np.ndarray = None # kiedy zdecydował o ucieczce
    gawking: np.ndarray = None  # bool: stoi / idzie na punkt obserwacji
    dist_event: np.ndarray = None   # odległość od zdarzenia w chwili jego wystąpienia
    dose: np.ndarray = None     # dawka urazu [N/m * s]
    pressure_s: np.ndarray = None   # ciśnienie wygładzone w czasie [N/m] (wejście modelu urazów)
    inj: np.ndarray = None      # int8: INJ_*
    knows: np.ndarray = None    # bool (n, K): znajomość bram
    gate_pen: np.ndarray = None # float32 (n, K): osobista kara za bramę, przy której utknął
    stuck_t: np.ndarray = None  # jak długo stoi w zatorze
    n_switch: np.ndarray = None # ile razy zmienił bramę
    returner: np.ndarray = None # bool: skłonny wrócić po znajomego z grupy
    returning: np.ndarray = None  # bool: właśnie idzie po znajomego
    mate: np.ndarray = None     # int32: po kogo idzie (-1 = nikt)
    t_return: np.ndarray = None # kiedy zaczął wracać
    redirected: np.ndarray = None  # bool: przekierowany przez służby

    def __post_init__(self):
        n = self.n
        z = lambda dt=float: np.zeros(n, dtype=dt)   # noqa: E731
        if self.axe is None:
            self.axe, self.aye = z(), z()
        if self.aw is None:
            self.aw = z(np.int8)
            self.t_notice = np.full(n, np.nan)
            self.t_decide = np.full(n, np.nan)
            self.gawking = z(bool)
            self.dist_event = np.full(n, np.nan)
            self.dose = z()
            self.pressure_s = z()
            self.inj = z(np.int8)
            self.stuck_t = z()
            self.n_switch = z(np.int32)
            self.returner = z(bool)
            self.returning = z(bool)
            self.mate = np.full(n, -1, dtype=np.int32)
            self.t_return = np.full(n, np.nan)
            self.redirected = z(bool)


def _truncnorm(rng, mean, sd, lo, hi, n):
    return np.clip(rng.normal(mean, sd, n), lo, hi)


def create_agents(scene, wf, cfg, rng: np.random.Generator, reach: np.ndarray | None = None,
                  rng2: np.random.Generator | None = None) -> Agents:
    """reach - maska komórek, z których da się dojść do otwartej bramy (None = bez ograniczenia).
    rng2 - osobny strumień losowy dla modułów rozszerzonych (nie zaburza losowania dotychczasowego)."""
    ac = cfg.agents
    N = int(ac.count)
    r = rng.uniform(ac.radius_m[0], ac.radius_m[1], N)
    free = scene.free_points(wf, float(ac.min_wall_dist_m), rng=rng, mask=reach)
    if len(free) == 0:
        raise RuntimeError("Brak wolnych punktów w obrysie - sprawdź mapę (czy jest otwarta brama?).")
    rng2 = rng2 or np.random.default_rng(0)

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
    # skupiska startowe przy grillach (i ławkach): część grup zaczyna w pobliżu
    cl = getattr(ac, "clusters", None)
    if cl is not None and float(cl.share) > 0:
        attr = [np.asarray(g["xy"], dtype=float) for g in scene.grills]
        wts = [1.0] * len(attr)
        if float(cl.benches_weight) > 0 and scene.benches:
            attr += [np.asarray(b["xy"], dtype=float) for b in scene.benches]
            wts += [float(cl.benches_weight)] * len(scene.benches)
        if attr:
            attr = np.array(attr)
            ftree = cKDTree(free)
            ok_a = ftree.query(attr)[0] < 3.0              # tylko atraktory w części aktywnej
            attr, wts = attr[ok_a], np.array(wts)[ok_a]
            if len(attr):
                pick = np.flatnonzero(rng2.random(n_groups) < float(cl.share))
                a_i = rng2.choice(len(attr), len(pick), p=wts / wts.sum())
                cand = attr[a_i] + rng2.normal(0.0, float(cl.sigma_m), (len(pick), 2))
                centers[pick] = free[ftree.query(cand)[1]]   # dociągnięcie do najbliższego wolnego punktu
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
        good = good & (wf.walk[iyc, ixc] > 0) & scene.inside[iyc, ixc] & (wf.d[iyc, ixc] >= r + 0.1)
        return good if reach is None else good & reach[iyc, ixc]

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
    n_gawk = int(round(float(bc.gawkers.share) * N))
    typ[perm[n_pan + n_ign: n_pan + n_ign + n_gawk]] = T_GAWKER

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
    # znajomość bram: każda osoba zna bramę k z prawdopodobieństwem knowledge[k]
    K = len(scene.exits)
    kn = dict(vars(cfg.exits.choice.knowledge))
    prob = np.array([float(kn.get(e.id, kn.get("default", 1.0))) for e in scene.exits])
    A.knows = rng2.random((N, K)) < prob[None, :]
    A.gate_pen = np.zeros((N, K), dtype=np.float32)
    A.returner = grouped & (rng2.random(N) < float(bc.group_return.share))
    return A
