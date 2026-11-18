"""Wybór wyjścia - model rozszerzony (exits.choice.mode = softmax).

  * pierwszy wybór losowy: P(brama) ~ exp(-koszt / temperature_m) - niekoniecznie najbliższa,
  * częściowa znajomość bram (np. W6 zna 15% osób); bramę można "odkryć": widząc ją z bliska albo
    naśladując sąsiadów, którzy do niej idą,
  * naśladowanie sąsiadów: brama wybierana przez osoby w pobliżu jest "tańsza" (silniej przy panice),
  * zator wykrywany LOKALNIE: agent, który długo stoi w tłoku daleko od bramy, dolicza swojej bramie karę
    i wybiera ponownie; kolejkę przy bramie zna tylko ten, kto ją widzi (`see_dist_m`),
  * stopniowa racjonalność: waga informacji o kolejkach = (1 - p)^gamma zamiast progu p = 0.6
    (z dolnym ograniczeniem `rationality_min`),
  * opcjonalnie: szersza brama jest atrakcyjniejsza (`width_bonus_m`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .fields import BIG


def _share_grid(sim):
    """Udział bram wśród uciekających w okolicy (siatka o boku = promień naśladowania, blok 3x3)."""
    A, sc = sim.A, sim.scene
    cell = float(sim.cfg.exits.choice.herding.radius_m)
    x0, y0, x1, y1 = sc.bounds
    gnx, gny = int((x1 - x0) / cell) + 2, int((y1 - y0) / cell) + 2
    cnt = np.zeros((sim.K, gny, gnx))
    mv = np.flatnonzero((A.state == 1) & (A.target >= 0))
    if len(mv):
        cx = np.clip(((A.x[mv] - x0) / cell).astype(int), 0, gnx - 1)
        cy = np.clip(((A.y[mv] - y0) / cell).astype(int), 0, gny - 1)
        np.add.at(cnt, (A.target[mv], cy, cx), 1.0)
        cnt = ndimage.uniform_filter(cnt, size=(1, 3, 3), mode="constant") * 9.0
    tot = cnt.sum(axis=0)
    share = cnt / np.maximum(tot, 1.0)

    def at(x, y):
        cx = np.clip(((x - x0) / cell).astype(int), 0, gnx - 1)
        cy = np.clip(((y - y0) / cell).astype(int), 0, gny - 1)
        return share[:, cy, cx]
    return at


def _totals(sim, idx, share_at, first):
    A, ch = sim.A, sim.cfg.exits.choice
    cost = sim.nav.cost_at(A.x[idx], A.y[idx])                      # (K, n)
    possible = (sim.open[:, None] == 1) & (cost < BIG / 2)
    feasible = possible & A.knows[idx].T
    none = ~feasible.any(axis=0)
    if none.any():          # nie zna żadnej czynnej bramy -> idzie do najbliższej czynnej (i ją poznaje)
        c2 = np.where(possible[:, none], cost[:, none], np.inf)
        kbest = np.argmin(c2, axis=0)
        A.knows[idx[none], kbest] = True
        feasible[kbest, np.flatnonzero(none)] = True
    p = A.p[idx]
    share = share_at(A.x[idx], A.y[idx])
    total = cost - float(ch.herding.weight_m) * (1.0 + p)[None, :] * share
    if float(ch.width_bonus_m) > 0:
        total = total - float(ch.width_bonus_m) * np.log(np.maximum(sim.widths, 0.5))[:, None]
    total = total - sim.gate_bonus[:, None]
    if not first:
        w_r = np.maximum((1.0 - p) ** float(ch.rationality_gamma), float(ch.rationality_min))
        visible = cost < float(ch.see_dist_m)
        total = total + w_r[None, :] * (np.minimum(sim.queue_pen, 1e6)[:, None] * visible) + A.gate_pen[idx].T
    total = np.where(feasible, total, np.inf)
    return total, cost, share, feasible


def _sample(sim, total, T):
    z = -(total - total.min(axis=0, keepdims=True)) / max(T, 1e-6)
    pr = np.exp(z)
    pr /= pr.sum(axis=0, keepdims=True)
    cum = np.cumsum(pr, axis=0)
    u = sim.rng2.random(total.shape[1])
    return np.minimum((cum < u[None, :]).sum(axis=0), total.shape[0] - 1)


def replan(sim, idx, force=False, tau=0.0):
    if len(idx) == 0:
        return
    A, ch = sim.A, sim.cfg.exits.choice
    T = float(ch.temperature_m)
    share_at = _share_grid(sim)
    new = A.target[idx] < 0
    if new.any():                                   # --- pierwszy wybór
        i_new = idx[new]
        total, _, _, _ = _totals(sim, i_new, share_at, first=True)
        A.target[i_new] = _sample(sim, total, T)
    rest = idx[~new]
    if len(rest) == 0:
        return
    lj, hd = ch.local_jam, ch.herding
    cur = A.target[rest]
    cols = np.arange(len(rest))
    cost = sim.nav.cost_at(A.x[rest], A.y[rest])
    # --- poznawanie bram: z bliska albo od sąsiadów
    share = share_at(A.x[rest], A.y[rest])
    learn = (cost < float(ch.see_dist_m)) | ((share > float(hd.learn_frac))
                                             & (sim.rng2.random(share.shape) < float(hd.learn_prob)))
    A.knows[rest] |= learn.T
    if not force:
        dtp = float(sim.cfg.exits.replan_s)
        slow = np.hypot(A.vx[rest], A.vy[rest]) < float(lj.slow_mps)
        A.stuck_t[rest] = np.where(slow, A.stuck_t[rest] + dtp, np.maximum(A.stuck_t[rest] - dtp, 0.0))
        A.gate_pen[rest] *= np.float32(np.exp(-dtp / float(lj.memory_s)))
    feasible_cur = (sim.open[cur] == 1) & (cost[cur, cols] < BIG / 2)
    # --- służby porządkowe: w zasięgu służb agent zna wszystkie bramy i może zostać przekierowany
    st = sim.steward_gates & (sim.open == 1)
    if st.any():
        sc_ = sim.cfg.stewards
        near = (cost[st] < float(sc_.radius_m)).any(axis=0)
        A.knows[rest[near]] = True
        # przekierowuje tylko służba przy bramie, do której agent idzie, i tylko raz (bez odsyłania tam i z powrotem)
        at_own = st[cur] & (cost[cur, cols] < float(sc_.radius_m)) & ~A.redirected[rest]
        cand = at_own & feasible_cur & (sim.rng2.random(len(rest)) < float(sc_.redirect_prob))
        if cand.any():
            ci = np.flatnonzero(cand)
            # służby mają łączność: znają obciążenie KAŻDEJ bramy = liczba osób, które do niej idą,
            # przeliczona na czas oczekiwania (przepływ j_ref * szerokość) i dalej na metry marszu
            ec = sim.cfg.exits
            load = np.bincount(A.target[(A.state == 1) & (A.target >= 0)], minlength=sim.K)
            wait_m = load / np.maximum(float(ec.j_ref) * sim.widths, 1e-6) * float(ec.walking_speed_mps)
            full = cost[:, ci] + wait_m[:, None] - sim.gate_bonus[:, None]
            full = np.where((sim.open[:, None] == 1) & (cost[:, ci] < BIG / 2), full, np.inf)
            best = np.argmin(full, axis=0)
            better = full[best, np.arange(len(ci))] < float(sim.cfg.exits.hysteresis) * full[cur[ci], np.arange(len(ci))]
            sw = ci[better & (best != cur[ci])]
            A.target[rest[sw]] = best[better & (best != cur[ci])]
            A.n_switch[rest[sw]] += 1
            A.redirected[rest[sw]] = True
            A.stuck_t[rest[sw]] = 0.0
            cur = A.target[rest]
    w_r = np.maximum((1.0 - A.p[rest]) ** float(ch.rationality_gamma), float(ch.rationality_min))
    stuck = feasible_cur & (A.stuck_t[rest] >= float(lj.patience_s)) & (cost[cur, cols] > float(lj.near_gate_m)) \
        & (sim.rng2.random(len(rest)) < w_r)
    A.gate_pen[rest[stuck], cur[stuck]] += np.float32(lj.penalty_m)
    A.stuck_t[rest[stuck]] = 0.0
    recon = stuck | ~feasible_cur
    if recon.any():
        i_re = rest[recon]
        total, _, _, _ = _totals(sim, i_re, share_at, first=False)
        newt = _sample(sim, total, 0.5 * T)
        A.n_switch[i_re] += (newt != cur[recon]) & feasible_cur[recon]
        A.target[i_re] = newt
