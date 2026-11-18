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

from . import alarm as alarm_mod
from . import choice as choice_mod
from . import injury as injury_mod
from .agents import (AW_FLEEING, AW_UNAWARE, EVACUATED, FALLEN, IDLE, INJ_LIGHT, INJ_SEVERE, MOVING, T_GAWKER,
                     T_IGNORER, T_NORMAL, T_PANICKER, Agents, create_agents)
from .engine import Physics
from .events import Events
from .fields import Nav, Smoke, SmokeAdv
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
    area: dict = field(default_factory=dict)        # pola: obrys / dostępne / aktywne / wyłączone [m2]
    events_obj: object = None                       # zdarzenia (pozycje do rysowania)
    heat: dict = field(default_factory=dict)        # mapy maks. gęstości i ciśnienia
    extra: dict = field(default_factory=dict)       # tablice pomocnicze (t_inj itd.)


class Simulation:
    def __init__(self, cfg, map_path: str, backend: str | None = None, progress=None, record=True):
        self.cfg = cfg
        self.progress = progress
        self.record = record
        self.rng = np.random.default_rng(int(cfg.seed))
        self.rng2 = np.random.default_rng([int(cfg.seed), 7])     # moduły rozszerzone (osobny strumień)
        self.scene = load_scene(map_path, cfg, extra_barriers=[dict(b) for b in (cfg.barriers or [])])
        sc = self.scene
        for w in sc.warnings:
            print("Uwaga (mapa):", w)
        self.PANICKER_P = PANICKER_P
        self.staged = str(cfg.alarm.mode) == "staged"
        self.softmax = str(cfg.exits.choice.mode) == "softmax"
        self.dose_mode = str(cfg.injury.mode) == "dose" and bool(cfg.injury.enabled)
        self.K = len(sc.exits)
        self.open = np.array([1 if o else 0 for o in sc.initial_open()], dtype=np.int8)
        if not self.open.any():
            raise RuntimeError("Wszystkie bramy są zamknięte - nie ma dokąd ewakuować.")
        self.wf = sc.wall_fields(self.open)
        # osiągalność: agenci tylko tam, skąd da się dojść do otwartej bramy (płot odcina część południową)
        self.reach = sc.reachable(self.wf, self.open) if bool(cfg.map_options.reachability) else None
        self.area = sc.area_report(self.wf, self.open)
        self.free_pts = sc.free_points(self.wf, 2.0, mask=self.reach)
        self.pending_log = []
        self.close_events = sorted(
            [(float(t), str(k)) for k, t in _as_dict(cfg.exits.close_at_s).items()])
        self.events = Events(self)
        self.smoke, self.smoke_start = self._make_smoke()
        self.t_nav = -1e9
        self.nav = Nav(sc, self.wf, self.open, cfg)
        self.A = create_agents(sc, self.wf, cfg, self.rng, reach=self.reach,
                               rng2=np.random.default_rng([int(cfg.seed), 5]))
        self.phys = Physics(sc, cfg, self.A, self.wf, self.nav, self.open, backend=backend)
        if float(cfg.agents.party_phase_s) > 0:
            self.phys.e_open[:] = 0      # w fazie imprezy nikt nie wychodzi przez bramy
        self.exit_ids = [e.id for e in sc.exits]
        self.centers = np.array([e.center for e in sc.exits])
        self.widths = np.array([e.width for e in sc.exits])
        self.alarm_done = False
        self.t_last_replan = -1e9
        self.t_last_rec = -1e9
        self.t_last_ts = -1e9
        self.queue_pen = np.zeros(self.K)
        self.res = Result(cfg=cfg, scene=sc, agents=self.A, exit_ids=self.exit_ids,
                          backend=self.phys.backend, party_s=float(cfg.agents.party_phase_s),
                          area=self.area, events_obj=self.events)
        self.t_global = None
        self.gawk_peak = 0
        self.hurry = 1.0                              # mnożnik prędkości pożądanej (burza)
        self.gate_bonus = np.zeros(self.K)            # premia bramy w metrach (zadaszone przejścia przy burzy)
        stw = cfg.stewards
        self.steward_gates = np.zeros(self.K, dtype=bool)
        if bool(stw.enabled):
            ids = self.exit_ids if stw.gates == "all" else [str(g) for g in stw.gates]
            for g in ids:
                if g in sc.exit_index:
                    self.steward_gates[sc.exit_index[g]] = True
        sc.steward_gates = self.steward_gates
        self.members = {}                             # grupa -> indeksy członków (tylko grupy >= 2)
        if self.A.returner.any():
            order = np.argsort(self.A.gid, kind="stable")
            bounds = np.flatnonzero(np.diff(self.A.gid[order])) + 1
            for part in np.split(order, bounds):
                if len(part) >= 2:
                    self.members[int(self.A.gid[part[0]])] = part
        self.t_inj = np.full(self.A.n, np.nan)
        self.res.extra["t_inj"] = self.t_inj
        # mapa cieplna (maks. gęstość i ciśnienie na siatce 2 m)
        self.heat_on = bool(cfg.output.heatmap)
        hc = 2.0
        x0, y0, x1, y1 = sc.bounds
        self.heat_edges = (np.arange(x0, x1 + hc, hc), np.arange(y0, y1 + hc, hc))
        shape = (len(self.heat_edges[1]) - 1, len(self.heat_edges[0]) - 1)
        self.res.heat = {"cell": hc, "x0": x0, "y0": y0, "dens": np.zeros(shape), "press": np.zeros(shape)}
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
        idle = np.flatnonzero((A.state == IDLE) & (A.mode == 0) & (A.aw == AW_UNAWARE))
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
    def _make_smoke(self):
        cfg, ev = self.cfg, self.events
        if not ev.fires:
            return None, 0.0
        if str(cfg.hazard.model) == "advection":
            src = [{"xy": f["xy"], "t": f["t"]} for f in ev.fires]
            return SmokeAdv(self.scene, cfg, src, rng=np.random.default_rng([int(cfg.seed), 11])), -1e9
        sm = Smoke(self.scene, cfg)
        if not ev.legacy:
            ox, oy = ev.fires[0]["xy"]
            sm.src = (int((oy - sm.y0) / sm.h), int((ox - sm.x0) / sm.h))
            return sm, float(ev.fires[0]["t"])
        return sm, float(cfg.hazard.start_s)

    def notice(self, idx, tau):
        alarm_mod.notice(self, idx, tau)

    def _alarm(self, tau):
        if self.staged:
            alarm_mod.start(self, tau)
            return
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
        ign_m = (A.typ == T_IGNORER) & (A.state == IDLE) & np.isinf(A.tmove)
        if self.staged:
            ign_m &= A.aw != AW_UNAWARE
        ign = np.flatnonzero(ign_m)
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
        if fb is not None and self.staged:      # bezpiecznik liczony od chwili zauważenia zdarzenia
            m = ign_m & (tau - A.t_notice >= float(fb))
            A.tmove[m] = tau
        elif fb is not None and tau >= float(fb):
            m = (A.state == IDLE) & np.isinf(A.tmove)
            A.tmove[m] = tau
        # --- ruszenie
        go = np.flatnonzero((A.state == IDLE) & (A.tmove <= tau))
        if len(go) and self.members:
            go = self._start_returns(go, tau)
        if len(go):
            A.state[go] = MOVING
            A.t_start[go] = tau
            A.mode[go] = 0
            A.aw[go] = AW_FLEEING
            A.gawking[go] = False
            A.returning[go] = False
            self._replan(go, force=True, tau=tau)

    # ------------------------------------------------------ wracanie po znajomych
    def _start_returns(self, go, tau):
        """Część ruszających (A.returner) idzie najpierw po kolegę z grupy, który stoi albo leży."""
        A, gr = self.A, self.cfg.behaviour.group_return
        keep = np.ones(len(go), dtype=bool)
        for n_, i in enumerate(go):
            if not A.returner[i] or not np.isnan(A.t_return[i]):
                continue
            mem = self.members.get(int(A.gid[i]))
            if mem is None:
                continue
            mem = mem[(mem != i) & ((A.state[mem] == IDLE) | (A.state[mem] == FALLEN)) & ~A.returning[mem]
                      & (A.tmove[mem] > tau + 3.0)]
            if len(mem) == 0:
                continue
            d = np.hypot(A.x[mem] - A.x[i], A.y[mem] - A.y[i])
            j = int(np.argmin(d))
            if not (3.0 < d[j] <= float(gr.max_dist_m)):
                continue
            keep[n_] = False
            A.returning[i] = True
            A.mate[i] = mem[j]
            A.t_return[i] = tau
            A.tmove[i] = np.inf
            A.mode[i] = 1
            A.v0eff[i] = A.v0[i]
            A.aw[i] = max(int(A.aw[i]), 2)
        return go[keep]

    def _return_step(self, tau):
        A, gr = self.A, self.cfg.behaviour.group_return
        A.returning &= A.state == IDLE
        idx = np.flatnonzero(A.returning)
        if len(idx) == 0:
            return
        m = A.mate[idx]
        A.wx[idx], A.wy[idx] = A.x[m], A.y[m]
        d = np.hypot(A.x[m] - A.x[idx], A.y[m] - A.y[idx])
        arrived = d < 2.5
        # dotarł: kolega (jeśli stoi) dowiaduje się i rusza razem z nim
        tell = m[arrived & (A.state[m] == IDLE)]
        if len(tell):
            A.gawking[tell] = False
            A.tmove[tell] = np.minimum(A.tmove[tell], tau + self.rng2.uniform(0.0, 2.0, len(tell)))
            fresh = tell[A.aw[tell] == AW_UNAWARE]
            A.aw[fresh] = 1
            A.t_notice[fresh] = tau
            A.mode[fresh] = 0
            A.v0eff[fresh] = 0.0
        wait = arrived & (A.state[m] == FALLEN)          # czeka przy leżącym, aż wstanie (albo zrezygnuje)
        A.v0eff[idx[wait]] = 0.0
        done = (arrived & ~wait) | (A.state[m] == MOVING) | (A.state[m] == EVACUATED) \
            | (tau - A.t_return[idx] >= float(gr.give_up_s))
        fin = idx[done]
        A.returning[fin] = False
        A.mode[fin] = 0
        A.v0eff[fin] = 0.0
        A.tmove[fin] = tau

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
        if self.softmax:
            choice_mod.replan(self, idx, force=force, tau=tau)
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
            self.close_gate(k, tau, f"zamknięto {eid}")

    def close_gate(self, k, tau, label):
        self.open[k] = 0
        self.wf = self.scene.wall_fields(self.open)
        self.nav = Nav(self.scene, self.wf, self.open, self.cfg, extra_cost=self._extra_cost_fn(tau))
        self.phys.set_fields(self.wf, self.nav, self.open)
        self.res.events.append((tau, label))
        mv = np.flatnonzero(self.A.state == MOVING)
        self._update_queue_penalty(tau)
        self._replan(mv, force=True, tau=tau)

    # ------------------------------------------------ zagrożenia a trasy i siły
    def _extra_cost_fn(self, tau):
        """Dodatkowy koszt trasy: dym (omijanie, strefa nie do wejścia) i strefy wykluczenia. None = brak."""
        hz = self.cfg.hazard
        sm = self.smoke if (isinstance(self.smoke, SmokeAdv) and float(hz.route_cost) > 0 and tau >= 0) else None
        zones = self.events.active_zones(tau)
        if sm is None and not zones:
            return None

        def fn(XC, YC):
            out = np.zeros_like(XC)
            if sm is not None:
                c = sm.sample(XC, YC)
                out += float(hz.route_cost) * np.clip(c, 0.0, 1.0) + float(hz.nogo_cost) * (c >= float(hz.nogo_conc))
            for z in zones:
                out += z["cost"] * (np.hypot(XC - z["xy"][0], YC - z["xy"][1]) <= z["R"])
            return out
        return fn

    def _hazard_update(self, tau):
        """Co `nav_update_s`: trasy z uwzględnieniem dymu/stref; co krok: przyspieszenie odpychające."""
        hz, A = self.cfg.hazard, self.A
        fn = self._extra_cost_fn(tau)
        if fn is None:
            return
        if tau - self.t_nav >= float(hz.nav_update_s):
            self.t_nav = tau
            self.nav = Nav(self.scene, self.wf, self.open, self.cfg, extra_cost=fn)
            self.phys.set_fields(self.wf, self.nav, self.open)
        A.axe[:] = 0.0
        A.aye[:] = 0.0
        mv = np.flatnonzero(A.state == MOVING)
        if len(mv) == 0:
            return
        if isinstance(self.smoke, SmokeAdv) and float(hz.repulsion_mps2) > 0:
            gx, gy = self.smoke.gradient(A.x[mv], A.y[mv])
            g = np.hypot(gx, gy)
            w = float(hz.repulsion_mps2) * np.clip(A.smoke[mv], 0.0, 1.0) / np.maximum(g, 1e-6) * (g > 1e-6)
            A.axe[mv] -= w * gx
            A.aye[mv] -= w * gy
        for z in self.events.active_zones(tau):
            dx, dy = A.x[mv] - z["xy"][0], A.y[mv] - z["xy"][1]
            d = np.maximum(np.hypot(dx, dy), 1e-6)
            w = z["push"] * np.clip((z["R"] + 1.0 - d) / 3.0, 0.0, 1.0) / d
            A.axe[mv] += w * dx
            A.aye[mv] += w * dy

    # ------------------------------------------------------- psychika i prędkość
    def _psychology(self, tau, dt):
        A, bc, hz = self.A, self.cfg.behaviour, self.cfg.hazard
        alive = np.flatnonzero(A.state != EVACUATED)
        if len(alive) == 0:
            return
        # dym
        if self.smoke is not None and tau >= self.smoke_start:
            if isinstance(self.smoke, SmokeAdv):
                self.smoke.step(dt, tau)
            else:
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
        if self.steward_gates.any():     # służby uspokajają osoby w swoim zasięgu
            stw = self.cfg.stewards
            c = self.centers[self.steward_gates & (self.open == 1)]
            if len(c):
                dmin = np.min(np.hypot(pos[:, None, 0] - c[None, :, 0], pos[:, None, 1] - c[None, :, 1]), axis=1)
                nearst = alive[dmin < float(stw.radius_m)]
                A.p[nearst] *= 1.0 - float(stw.calm_per_s) * dt
        pan = alive[A.typ[alive] == T_PANICKER]
        A.p[pan] = np.maximum(A.p[pan], PANICKER_P)
        # prędkość pożądana
        mv = np.flatnonzero(A.state == MOVING)
        if len(mv):
            base = A.v0[mv] * (1 - A.p[mv]) + float(bc.v_max_mps) * A.p[mv]
            slow = 1.0 - float(hz.slow_factor) * np.clip(A.smoke[mv], 0, 1)
            A.v0eff[mv] = base * np.maximum(slow, 0.3) * self.hurry
            ds = bc.density_speed
            if bool(ds.enabled):
                # diagram fundamentalny (Weidmann): v(rho) = v0 * (1 - exp(-gamma * (1/rho - 1/rho_max)))
                R = float(ds.radius_m)
                cnt = tree.query_ball_point(np.column_stack((A.x[mv], A.y[mv])), r=R, return_length=True)
                rho = np.maximum(cnt, 1) / (math.pi * R * R)
                f = 1.0 - np.exp(-float(ds.gamma) * (1.0 / rho - 1.0 / float(ds.rho_max)))
                A.v0eff[mv] *= np.clip(f, float(ds.min_factor), 1.0)
            if self.dose_mode:
                A.v0eff[mv] *= np.where(A.inj[mv] == INJ_LIGHT, float(self.cfg.injury.light_speed_factor), 1.0)
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
        if self.dose_mode:
            injury_mod.step(self, tau, dt)
            return
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

    # ------------------------------------------------------------ osoby w ścianie
    def _unstick(self):
        """Osoba, która znalazła się w komórce niedostępnej (ściana, szczelina), jest przesuwana do najbliższego
        wolnego miejsca. Bez tego tkwi w ścianie, sprężyna kontaktu daje wielokrotnie nierealne ciśnienie
        (rzędu 10 kN/m) i taka osoba zawyża liczbę urazów."""
        if not bool(getattr(self.cfg.map_options, "unstick", True)):
            return
        wf, sc, A = self.wf, self.scene, self.A
        if getattr(self, "_unstick_wf", None) is not wf:
            from scipy import ndimage
            good = (wf.walk > 0) & (wf.d >= 0.45)
            _, ind = ndimage.distance_transform_edt(~good, return_indices=True)
            self._unstick_ind, self._unstick_wf = ind, wf
        live = np.flatnonzero(A.state != EVACUATED)
        if not len(live):
            return
        ix = np.floor((A.x[live] - sc.x0) / sc.cw).astype(int)
        iy = np.floor((A.y[live] - sc.y0) / sc.cw).astype(int)
        ok = (ix >= 0) & (ix < sc.nx) & (iy >= 0) & (iy < sc.ny)
        bad = ~ok
        bad[ok] = wf.walk[iy[ok], ix[ok]] == 0
        if not bad.any():
            return
        idx = live[bad]
        ci = np.clip(ix[bad], 0, sc.nx - 1)
        cj = np.clip(iy[bad], 0, sc.ny - 1)
        nj, ni = self._unstick_ind[0][cj, ci], self._unstick_ind[1][cj, ci]
        A.x[idx], A.y[idx] = sc.xs[ni], sc.ys[nj]
        A.vx[idx] = 0.0
        A.vy[idx] = 0.0
        self.n_unstuck = getattr(self, "n_unstuck", 0) + len(idx)

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
            "unaware": int(((A.aw == AW_UNAWARE) & (st == IDLE)).sum()) if self.staged else 0,
            "gawkers": int(A.gawking.sum()),
            "returning": int(A.returning.sum()),
            "injured_light": int((A.inj == INJ_LIGHT).sum()),
            "injured_severe": int((A.inj == INJ_SEVERE).sum()),
            "mean_p": float(A.p[st != EVACUATED].mean()) if (st != EVACUATED).any() else 0.0,
            "max_pressure_Npm": float((A.pressure_s if self.dose_mode else A.pressure)[(st == MOVING) | (st == IDLE)].max())
            if ((st == MOVING) | (st == IDLE)).any() else 0.0,          # wygładzone (EMA, pressure_tau_s)
            "max_pressure_raw_Npm": float(A.pressure[(st == MOVING) | (st == IDLE)].max())
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
        if self.heat_on and len(live):
            ex, ey = self.heat_edges
            h = self.res.heat
            cnt, _, _ = np.histogram2d(A.y[live], A.x[live], bins=(ey, ex))
            np.maximum(h["dens"], cnt / h["cell"] ** 2, out=h["dens"])
            iy = np.clip(((A.y[live] - ey[0]) / h["cell"]).astype(int), 0, h["press"].shape[0] - 1)
            ix = np.clip(((A.x[live] - ex[0]) / h["cell"]).astype(int), 0, h["press"].shape[1] - 1)
            np.maximum.at(h["press"], (iy, ix), A.pressure[live])

    def _frame(self, tau):
        A = self.A
        fr = {
            "t": tau,
            "x": A.x.astype(np.float32).copy(), "y": A.y.astype(np.float32).copy(),
            "state": A.state.copy(), "p": (A.p * 255).astype(np.uint8),
            "open": self.open.copy(),
            "aw": A.aw.copy(), "gawk": A.gawking.copy() | A.returning, "inj": A.inj.copy(),
            "zones": [float(z["R"]) if tau >= z["t"] else 0.0 for z in self.events.zones],
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
        self.res.events.extend(self.pending_log)
        replan_dt = float(cfg.exits.replan_s)
        while True:
            # ---- część "wolna"
            if tau >= 0 and not self.alarm_done:
                self._alarm(tau)
            if self.alarm_done:
                self.events.step(tau, cdt)
                self._closures(tau)
                if self.staged:
                    alarm_mod.step(self, tau, cdt)
                    if bool(cfg.alarm.keep_wandering):
                        self._party_wander(tau)
                self._activate(tau)
                if self.members:
                    self._return_step(tau)
                self._psychology(tau, cdt)
                self._hazard_update(tau)
                if tau - self.t_last_replan >= replan_dt:
                    self._update_queue_penalty(tau)
                    mv = np.flatnonzero(self.A.state == MOVING)
                    self._replan(mv, tau=tau)
                    self.t_last_replan = tau
            else:
                self._party_wander(tau)
            self._unstick()
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
            busy = (st == IDLE) | (st == MOVING)
            if self.dose_mode:      # leżący, którzy jeszcze mogą wstać, też "trwają"
                busy = busy | ((st == FALLEN) & (self.A.inj < INJ_SEVERE))
            if self.alarm_done and not busy.any():
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
