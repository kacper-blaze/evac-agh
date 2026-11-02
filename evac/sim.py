"""Pętla główna symulacji.

Zegar `tau` liczony jest OD ALARMU: przed alarmem tau < 0 (faza "imprezy", długość = agents.party_phase_s).
Fizyka (SFM) liczy się co `dt_s`; "wolne" rzeczy (zachowanie, panika, wybór bramy, dym, statystyki)
co `coarse_dt_s`.

Zachowanie po alarmie:
  * zwykli    - ruszają po losowym opóźnieniu `react_delay_s`,
  * panikujący (od początku) - ruszają natychmiast, mają p=0.8 i zarażają innych,
  * ignorujący - NIE ruszają z powodu alarmu; ruszają, gdy większość sąsiadów (promień
    `neighbor_radius_m`) jest już w ruchu. Bezpiecznik `ignore_fallback_s` (null = wyłączony).
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .agents import (EVACUATED, FALLEN, IDLE, MOVING, T_IGNORER, T_NORMAL, T_PANICKER, Agents,
                     create_agents)
from .engine import Physics
from .fields import Nav, Smoke
from .geometry import load_scene

PANICKER_P = 0.8      # poziom paniki agentów panikujących od początku
PANIC_SOURCE = 0.3    # od tego p agent "zaraża" innych


def _as_dict(x):
    if not x:
        return {}
    return dict(vars(x)) if not isinstance(x, dict) else x


@dataclass
class Result:
    cfg: object
    scene: object
    agents: Agents
    exit_ids: list
    ts: list = field(default_factory=list)          # wiersze statystyk czasowych
    frames: list = field(default_factory=list)      # klatki do wideo
    events: list = field(default_factory=list)      # zdarzenia (zamknięcia bram itp.)
    t_end: float = 0.0
    wall_time_s: float = 0.0
    backend: str = ""
    party_s: float = 0.0
    open_final: list = field(default_factory=list)


class Simulation:
    def __init__(self, cfg, map_path: str, backend: str | None = None, progress=None, record=True):
        self.cfg = cfg
        self.progress = progress
        self.record = record
        self.rng = np.random.default_rng(int(cfg.seed))
        self.scene = load_scene(map_path, cfg, extra_barriers=[dict(b) for b in (cfg.barriers or [])])
        sc = self.scene
        self.open = np.array([1 if o else 0 for o in sc.initial_open()], dtype=np.int8)
        self.wf = sc.wall_fields(self.open)
        self.nav = Nav(sc, self.wf, self.open, cfg)
        self.smoke = Smoke(sc, cfg) if cfg.hazard.enabled else None
        self.A = create_agents(sc, self.wf, cfg, self.rng)
        self.phys = Physics(sc, cfg, self.A, self.wf, self.nav, self.open, backend=backend)
        self.K = len(sc.exits)
        if float(cfg.agents.party_phase_s) > 0:
            self.phys.e_open[:] = 0      # w fazie imprezy nikt nie wychodzi przez bramy
        self.exit_ids = [e.id for e in sc.exits]
        self.centers = np.array([e.center for e in sc.exits])
        self.widths = np.array([e.width for e in sc.exits])
        self.alarm_done = False
        self.t_last_replan = -1e9
        self.t_last_rec = -1e9
        self.t_last_ts = -1e9
        self.close_events = sorted(
            [(float(t), str(k)) for k, t in _as_dict(cfg.exits.close_at_s).items()])
        self.queue_pen = np.zeros(self.K)
        self.res = Result(cfg=cfg, scene=sc, agents=self.A, exit_ids=self.exit_ids,
                          backend=self.phys.backend, party_s=float(cfg.agents.party_phase_s))
        self.peak_dens = np.zeros(self.K)
        self.max_press = 0.0
        self.t_down = np.full(self.A.n, np.nan)
        # stała siła spójności grup (tylko dla osób w grupach)
        sizes = np.bincount(self.A.gid, minlength=self.A.n_groups)
        self.grouped = sizes[self.A.gid] >= 2
        self.group_sizes = sizes

    # ------------------------------------------------------------------ impreza
    def _party_wander(self, tau):
        A = self.A
        sc = self.scene
        idle = np.flatnonzero((A.state == IDLE) & (A.mode == 0))
        if len(idle):
            pick = idle[self.rng.random(len(idle)) < 0.05]
            if len(pick):
                cand_x = A.x[pick] + self.rng.uniform(-6, 6, len(pick))
                cand_y = A.y[pick] + self.rng.uniform(-6, 6, len(pick))
                iy, ix = sc.cell_of(cand_x, cand_y)
                good = (iy >= 0) & (iy < sc.ny) & (ix >= 0) & (ix < sc.nx)
                iyc, ixc = np.clip(iy, 0, sc.ny - 1), np.clip(ix, 0, sc.nx - 1)
                good &= sc.inside[iyc, ixc] & (self.wf.walk[iyc, ixc] > 0) & (self.wf.d[iyc, ixc] >= 0.6)
                sel = pick[good]
                A.wx[sel] = cand_x[good]
                A.wy[sel] = cand_y[good]
                A.mode[sel] = 1
                A.v0eff[sel] = A.vw[sel]
        walking = np.flatnonzero((A.state == IDLE) & (A.mode == 1))
        if len(walking):
            d = np.hypot(A.wx[walking] - A.x[walking], A.wy[walking] - A.y[walking])
            arrived = walking[d < 0.4]
            A.mode[arrived] = 0
            A.v0eff[arrived] = 0.0

    # --------------------------------------------------------------------- alarm
    def _alarm(self, tau):
        A, bc = self.A, self.cfg.behaviour
        A.mode[:] = 0
        A.v0eff[:] = 0.0
        lo, hi = bc.react_delay_s
        plo, phi = bc.panic_delay_s
        n = A.n
        A.tmove[:] = tau + self.rng.uniform(lo, hi, n)
        pan = A.typ == T_PANICKER
        A.tmove[pan] = tau + self.rng.uniform(plo, phi, int(pan.sum()))
        A.p[pan] = PANICKER_P
        A.tmove[A.typ == T_IGNORER] = np.inf
        self._ignore_ready = np.zeros(n, dtype=bool)
        self.alarm_done = True
        self.phys.e_open[:] = self.open
        self.res.events.append((tau, "alarm"))

    # ------------------------------------------------------------------- aktywacja
    def _activate(self, tau):
        A, bc = self.A, self.cfg.behaviour
        alive = np.flatnonzero(A.state != EVACUATED)
        # --- ignorujący: reguła większości sąsiadów
        ign = np.flatnonzero((A.typ == T_IGNORER) & (A.state == IDLE) & np.isinf(A.tmove))
        if len(ign):
            tree = cKDTree(np.column_stack((A.x[alive], A.y[alive])))
            moving_alive = (A.state[alive] == MOVING)
            pts = np.column_stack((A.x[ign], A.y[ign]))
            lists = tree.query_ball_point(pts, r=float(bc.neighbor_radius_m))
            self_pos = {int(a): k for k, a in enumerate(alive)}
            for q, lst in zip(ign, lists):
                me = self_pos[int(q)]
                nb = [j for j in lst if j != me]
                if not nb:
                    continue
                frac = float(moving_alive[nb].mean())
                if frac >= float(bc.majority):
                    A.tmove[q] = tau + self.rng.uniform(0.0, 2.0)
        fb = bc.ignore_fallback_s
        if fb is not None and tau >= float(fb):
            m = (A.state == IDLE) & np.isinf(A.tmove)
            A.tmove[m] = tau
        # --- ruszenie
        go = np.flatnonzero((A.state == IDLE) & (A.tmove <= tau))
        if len(go):
            A.state[go] = MOVING
            A.t_start[go] = tau
            A.mode[go] = 0
            self._replan(go, force=True, tau=tau)

    # --------------------------------------------------------------- planowanie bram
    def _update_queue_penalty(self, tau):
        ec, A = self.cfg.exits, self.A
        mv = np.flatnonzero(A.state == MOVING)
        pen = np.zeros(self.K)
        if len(mv):
            tree = cKDTree(np.column_stack((A.x[mv], A.y[mv])))
            nq = np.array(tree.query_ball_point(self.centers, r=float(ec.queue_radius_m), return_length=True),
                          dtype=float)
        else:
            nq = np.zeros(self.K)
        cap = float(ec.j_ref) * self.widths
        wait_s = nq / np.maximum(cap, 1e-6)
        pen = wait_s * float(ec.walking_speed_mps)
        recent = np.array([np.sum((A.exit_k == k) & (A.t_exit > tau - 5.0)) for k in range(self.K)]) / 5.0
        jam = (nq >= float(ec.jam_agents)) & (recent < float(ec.jam_flow_frac) * cap)
        pen = pen + jam * float(ec.jam_penalty_m)
        pen[self.open == 0] = 1e9
        self.queue_pen = pen
        self.nq = nq

    def _replan(self, idx, force=False, tau=0.0):
        if len(idx) == 0:
            return
        A, ec = self.A, self.cfg.exits
        cost = self.nav.cost_at(A.x[idx], A.y[idx])                 # (K, n)
        rational = (A.p[idx] < 0.6).astype(float)
        total = cost + self.queue_pen[:, None] * rational[None, :]
        closed = self.open == 0
        total[closed, :] = 1e12
        best = np.argmin(total, axis=0)
        cur = A.target[idx]
        have = cur >= 0
        cur_c = np.where(have, cur, 0)
        cur_cost = total[cur_c, np.arange(len(idx))]
        cur_ok = have & (self.open[cur_c] == 1)
        hyst = float(ec.hysteresis)
        switch = (~cur_ok) | (total[best, np.arange(len(idx))] < hyst * cur_cost)
        newt = np.where(switch, best, cur)
        A.target[idx] = newt

    # ------------------------------------------------------------- zamykanie bram
    def _closures(self, tau):
        while self.close_events and self.close_events[0][0] <= tau:
            t, eid = self.close_events.pop(0)
            if eid not in self.scene.exit_index:
                continue
            k = self.scene.exit_index[eid]
            if self.open[k] == 0:
                continue
            self.open[k] = 0
            self.wf = self.scene.wall_fields(self.open)
            self.nav = Nav(self.scene, self.wf, self.open, self.cfg)
            self.phys.set_fields(self.wf, self.nav, self.open)
            self.res.events.append((tau, f"zamknięto {eid}"))
            mv = np.flatnonzero(self.A.state == MOVING)
            self._update_queue_penalty(tau)
            self._replan(mv, force=True, tau=tau)

    # ------------------------------------------------------- psychika i prędkość
    def _psychology(self, tau, dt):
        A, bc, hz = self.A, self.cfg.behaviour, self.cfg.hazard
        alive = np.flatnonzero(A.state != EVACUATED)
        if len(alive) == 0:
            return
        # dym
        if self.smoke is not None and tau >= float(hz.start_s):
            self.smoke.step(dt)
            A.smoke[alive] = self.smoke.sample(A.x[alive], A.y[alive])
            A.p[alive] += float(hz.panic_rate_per_s) * A.smoke[alive] * dt / A.rho[alive] * (1 - A.p[alive])
        # zarażanie paniką
        ct = bc.contagion
        pos = np.column_stack((A.x[alive], A.y[alive]))
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=float(ct.radius_m), output_type="ndarray")
        S = np.zeros(len(alive))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            pi_, pj_ = A.p[alive][i], A.p[alive][j]
            wi = np.where(pj_ > PANIC_SOURCE, pj_, 0.0)
            wj = np.where(pi_ > PANIC_SOURCE, pi_, 0.0)
            S += np.bincount(i, weights=wi, minlength=len(alive))
            S += np.bincount(j, weights=wj, minlength=len(alive))
        pa = A.p[alive]
        dp = float(ct.rate_per_s) * np.minimum(S / float(ct.saturation), 1.0) * (1 - pa) / A.rho[alive]
        dp -= float(ct.decay_per_s) * pa
        A.p[alive] = np.clip(pa + dp * dt, 0.0, 1.0)
        pan = alive[A.typ[alive] == T_PANICKER]
        A.p[pan] = np.maximum(A.p[pan], PANICKER_P)
        # prędkość pożądana
        mv = np.flatnonzero(A.state == MOVING)
        if len(mv):
            base = A.v0[mv] * (1 - A.p[mv]) + float(bc.v_max_mps) * A.p[mv]
            slow = 1.0 - float(hz.slow_factor) * np.clip(A.smoke[mv], 0, 1)
            A.v0eff[mv] = base * np.maximum(slow, 0.3)
        # spójność grup (słabnie z paniką)
        gc = bc.group_cohesion
        live = (A.state != EVACUATED) & (A.state != FALLEN)
        w = live.astype(float)
        cnt = np.bincount(A.gid, weights=w, minlength=A.n_groups)
        sx = np.bincount(A.gid, weights=A.x * w, minlength=A.n_groups)
        sy = np.bincount(A.gid, weights=A.y * w, minlength=A.n_groups)
        safe = np.maximum(cnt, 1)
        A.cx[:] = (sx / safe)[A.gid]
        A.cy[:] = (sy / safe)[A.gid]
        ok = self.grouped & live & (cnt[A.gid] >= 2)
        A.coh[:] = np.where(ok, float(gc.gain) * (1 - A.p), 0.0)
        # upadek
        down = A.state == FALLEN
        new_down = down & np.isnan(self.t_down)
        self.t_down[new_down] = tau
        A.t_fall[new_down & np.isnan(A.t_fall)] = tau
        rec = self.cfg.injury.recover_s
        if rec is not None:
            up = down & (tau - self.t_down >= float(rec))
            A.state[up] = MOVING
            A.press_t[up] = 0.0
            self.t_down[up] = np.nan

    # --------------------------------------------------------------- statystyki
    def _stats(self, tau):
        A = self.A
        st = A.state
        row = {
            "t_s": tau,
            "evacuated": int((st == EVACUATED).sum()),
            "moving": int((st == MOVING).sum()),
            "idle": int((st == IDLE).sum()),
            "fallen": int((st == FALLEN).sum()),
            "mean_p": float(A.p[st != EVACUATED].mean()) if (st != EVACUATED).any() else 0.0,
            "max_pressure_Npm": float(A.pressure[(st == MOVING) | (st == IDLE)].max())
            if ((st == MOVING) | (st == IDLE)).any() else 0.0,
        }
        live = np.flatnonzero(st != EVACUATED)
        if len(live):
            tree = cKDTree(np.column_stack((A.x[live], A.y[live])))
            n2 = np.array(tree.query_ball_point(self.centers, r=2.0, return_length=True), dtype=float)
        else:
            n2 = np.zeros(self.K)
        dens = n2 / (math.pi * 4.0)
        self.peak_dens = np.maximum(self.peak_dens, dens)
        self.max_press = max(self.max_press, row["max_pressure_Npm"])
        for k, eid in enumerate(self.exit_ids):
            row[f"out_{eid}"] = int(((A.exit_k == k) & (st == EVACUATED)).sum())
            row[f"dens2m_{eid}"] = float(dens[k])
        self.res.ts.append(row)

    def _frame(self, tau):
        A = self.A
        fr = {
            "t": tau,
            "x": A.x.astype(np.float32).copy(), "y": A.y.astype(np.float32).copy(),
            "state": A.state.copy(), "p": (A.p * 255).astype(np.uint8),
            "open": self.open.copy(),
        }
        if self.smoke is not None:
            fr["smoke"] = self.smoke.snapshot_u8()
        self.res.frames.append(fr)

    # ----------------------------------------------------------------- uruchomienie
    def run(self):
        cfg = self.cfg
        dt = float(cfg.sim.dt_s)
        cdt = float(cfg.sim.coarse_dt_s)
        nsub = max(1, int(round(cdt / dt)))
        party = float(cfg.agents.party_phase_s)
        tmax = float(cfg.sim.t_max_s)
        tau = -party
        wall0 = _time.time()
        last_print = wall0
        self.res.events.append((tau, "start (faza imprezy)" if party > 0 else "start"))
        replan_dt = float(cfg.exits.replan_s)
        while True:
            # ---- część "wolna"
            if tau >= 0 and not self.alarm_done:
                self._alarm(tau)
            if self.alarm_done:
                self._closures(tau)
                self._activate(tau)
                self._psychology(tau, cdt)
                if tau - self.t_last_replan >= replan_dt:
                    self._update_queue_penalty(tau)
                    mv = np.flatnonzero(self.A.state == MOVING)
                    self._replan(mv, tau=tau)
                    self.t_last_replan = tau
            else:
                self._party_wander(tau)
            if tau - self.t_last_ts >= 1.0 - 1e-9:
                self._stats(tau)
                self.t_last_ts = tau
            if self.record and tau - self.t_last_rec >= float(cfg.sim.record_dt_s) - 1e-9:
                self._frame(tau)
                self.t_last_rec = tau
            if self.progress and _time.time() - last_print > 5.0:
                last_print = _time.time()
                st = self.A.state
                self.progress(tau, int((st == EVACUATED).sum()), int((st == MOVING).sum()),
                              int((st == FALLEN).sum()), _time.time() - wall0)
            # ---- warunki końca
            st = self.A.state
            if self.alarm_done and not ((st == IDLE) | (st == MOVING)).any():
                break
            if tau >= tmax:
                break
            # ---- część "szybka": fizyka
            for _ in range(nsub):
                self.phys.step(tau)
                tau += dt
        self._stats(tau)
        if self.record:
            self._frame(tau)
        self.res.t_end = tau
        self.res.wall_time_s = _time.time() - wall0
        self.res.backend = self.phys.backend
        self.res.open_final = [int(o) for o in self.open]
        return self.res
