"""Rdzeń fizyczny: Social Force Model (Helbing, Farkas, Vicsek 2000).

Dwa warianty tej samej fizyki:
  * `step_loop`  - zwykłe pętle; kompilowane przez Numbę (@njit), jeśli jest zainstalowana,
  * `step_numpy` - wektorowy NumPy + cKDTree (zapas, gdy Numba nie działa).

Siły na agenta i:
  napęd        m (v0 * e - v) / tau
  agent-agent  [A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij)] n_ij + kappa g(r_ij - d_ij) (v_j - v_i).t_ij t_ij
  agent-ściana to samo (ściana nieruchoma)
  grupa        słaba sprężyna do środka grupy (spójność)
Ciśnienie = suma fizycznych sił nacisku (k g) / obwód 2 pi r  (Helbing 2000: uraz > 1600 N/m).
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
from scipy.spatial import cKDTree

try:  # Numba jest opcjonalna
    if os.environ.get("EVAC_NO_JIT") == "1":
        raise ImportError("wyłączone przez EVAC_NO_JIT=1")
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f
        return deco

# indeksy w wektorze parametrów P
P_TAU, P_A, P_B, P_K, P_KAPPA, P_WA, P_WB, P_VCAP, P_COHFREE, P_PTHR, P_HOLD, P_INJ, P_GAP = range(13)


def pack_params(cfg) -> np.ndarray:
    s = cfg.sfm
    P = np.zeros(13)
    P[P_TAU] = s.tau_s
    P[P_A] = s.A_N
    P[P_B] = s.B_m
    P[P_K] = s.k
    P[P_KAPPA] = s.kappa
    P[P_WA] = s.wall_A_N
    P[P_WB] = s.wall_B_m
    P[P_VCAP] = cfg.engine.speed_cap_mps
    P[P_COHFREE] = cfg.behaviour.group_cohesion.free_dist_m
    P[P_PTHR] = cfg.injury.threshold_Npm
    P[P_HOLD] = cfg.injury.hold_s
    P[P_INJ] = 1.0 if cfg.injury.enabled else 0.0
    P[P_GAP] = 0.8   # powyżej tej odległości między powierzchniami siły są pomijalne (A exp(-0.8/0.08) ~ 0.1 N)
    return P


# ----------------------------------------------------------------------------- wariant: pętle (Numba)
@njit(cache=True)
def _bilerp(a, x, y, ox, oy, cs, nx, ny):
    fx = (x - ox) / cs - 0.5
    fy = (y - oy) / cs - 0.5
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    tx = fx - i0
    ty = fy - j0
    ia = min(max(i0, 0), nx - 1)
    ib = min(max(i0 + 1, 0), nx - 1)
    ja = min(max(j0, 0), ny - 1)
    jb = min(max(j0 + 1, 0), ny - 1)
    v0 = a[ja, ia] * (1.0 - tx) + a[ja, ib] * tx
    v1 = a[jb, ia] * (1.0 - tx) + a[jb, ib] * tx
    return v0 * (1.0 - ty) + v1 * ty


@njit(cache=True)
def step_loop(x, y, vx, vy, r, m, state, mode, v0eff, target, wx, wy, cx, cy, coh,
              press_t, pressure, exit_k, t_exit,
              wall_nbx, wall_nby, wall_walk, wox, woy, wcw, wnx, wny,
              nav_ex, nav_ey, nox, noy, ncs, nnx, nny,
              e_ax, e_ay, e_ux, e_uy, e_nx, e_ny, e_len, e_open,
              P, dt, tnow,
              head, nxt, gox, goy, gcs, gnx, gny, fx, fy, pn):
    n = x.shape[0]
    K = e_ax.shape[0]
    tau = P[0]
    A = P[1]
    B = P[2]
    kk = P[3]
    kappa = P[4]
    wA = P[5]
    wB = P[6]
    vcap = P[7]
    cfree = P[8]
    pthr = P[9]
    hold = P[10]
    inj = P[11]
    gap = P[12]
    two_pi = 2.0 * math.pi

    for c in range(gnx * gny):
        head[c] = -1
    for i in range(n):
        nxt[i] = -1
        if state[i] == 2:
            continue
        ci = int((x[i] - gox) / gcs)
        cj = int((y[i] - goy) / gcs)
        ci = min(max(ci, 0), gnx - 1)
        cj = min(max(cj, 0), gny - 1)
        c = cj * gnx + ci
        nxt[i] = head[c]
        head[c] = i

    for i in range(n):
        fx[i] = 0.0
        fy[i] = 0.0
        pn[i] = 0.0
        s = state[i]
        if s == 2 or s == 3:
            continue
        xi = x[i]
        yi = y[i]
        vxi = vx[i]
        vyi = vy[i]
        ri = r[i]
        mi = m[i]

        # --- napęd
        ex = 0.0
        ey = 0.0
        if s == 1:
            k = target[i]
            if k >= 0:
                ex = _bilerp(nav_ex[k], xi, yi, nox, noy, ncs, nnx, nny)
                ey = _bilerp(nav_ey[k], xi, yi, nox, noy, ncs, nnx, nny)
                nrm = math.sqrt(ex * ex + ey * ey)
                if nrm > 1.0e-6:
                    ex = ex / nrm
                    ey = ey / nrm
                else:
                    ex = 0.0
                    ey = 0.0
        elif mode[i] == 1:
            dxw = wx[i] - xi
            dyw = wy[i] - yi
            dd = math.sqrt(dxw * dxw + dyw * dyw)
            if dd > 0.3:
                ex = dxw / dd
                ey = dyw / dd
        fxi = mi * (v0eff[i] * ex - vxi) / tau
        fyi = mi * (v0eff[i] * ey - vyi) / tau
        pni = 0.0

        # --- inni agenci (lista komórek)
        ci = min(max(int((xi - gox) / gcs), 0), gnx - 1)
        cj = min(max(int((yi - goy) / gcs), 0), gny - 1)
        for dj in range(-1, 2):
            jj = cj + dj
            if jj < 0 or jj >= gny:
                continue
            for di in range(-1, 2):
                ii = ci + di
                if ii < 0 or ii >= gnx:
                    continue
                j = head[jj * gnx + ii]
                while j != -1:
                    if j != i:
                        dx = xi - x[j]
                        dy = yi - y[j]
                        d2 = dx * dx + dy * dy
                        rij = ri + r[j]
                        lim = rij + gap
                        if d2 < lim * lim:
                            d = math.sqrt(d2)
                            nxv = 1.0
                            nyv = 0.0
                            if d < 1.0e-9:
                                d = 1.0e-9
                            else:
                                nxv = dx / d
                                nyv = dy / d
                            g = rij - d
                            arg = g / B
                            if arg > 10.0:
                                arg = 10.0
                            gp = g if g > 0.0 else 0.0
                            fn = A * math.exp(arg) + kk * gp
                            fxi += fn * nxv
                            fyi += fn * nyv
                            tx = -nyv
                            ty = nxv
                            dvt = (vx[j] - vxi) * tx + (vy[j] - vyi) * ty
                            ft = kappa * gp * dvt
                            fxi += ft * tx
                            fyi += ft * ty
                            pni += kk * gp
                    j = nxt[j]

        # --- ściana
        iyw = int((yi - woy) / wcw)
        ixw = int((xi - wox) / wcw)
        if iyw >= 0 and iyw < wny and ixw >= 0 and ixw < wnx:
            bx = wall_nbx[iyw, ixw]
            by = wall_nby[iyw, ixw]
            dx = xi - bx
            dy = yi - by
            d = math.sqrt(dx * dx + dy * dy)
            ds = d - 0.5 * wcw
            if d > 1.0e-9 and (ds - ri) < gap:
                nxv = dx / d
                nyv = dy / d
                g = ri - ds
                arg = g / wB
                if arg > 10.0:
                    arg = 10.0
                gp = g if g > 0.0 else 0.0
                fn = wA * math.exp(arg) + kk * gp
                fxi += fn * nxv
                fyi += fn * nyv
                tx = -nyv
                ty = nxv
                vt = vxi * tx + vyi * ty
                ft = -kappa * gp * vt
                fxi += ft * tx
                fyi += ft * ty
                pni += kk * gp

        # --- spójność grupy
        if coh[i] > 0.0:
            dxg = cx[i] - xi
            dyg = cy[i] - yi
            dg = math.sqrt(dxg * dxg + dyg * dyg)
            if dg > cfree:
                mag = coh[i] * min(dg - cfree, 3.0) * mi
                fxi += mag * dxg / dg
                fyi += mag * dyg / dg

        fx[i] = fxi
        fy[i] = fyi
        pn[i] = pni

    # --- całkowanie, wyjścia, urazy
    for i in range(n):
        s = state[i]
        if s == 2 or s == 3:
            continue
        vxn = vx[i] + fx[i] / m[i] * dt
        vyn = vy[i] + fy[i] / m[i] * dt
        sp = math.sqrt(vxn * vxn + vyn * vyn)
        if sp > vcap:
            vxn = vxn * vcap / sp
            vyn = vyn * vcap / sp
        xn = x[i] + vxn * dt
        yn = y[i] + vyn * dt
        iyw = int((yn - woy) / wcw)
        ixw = int((xn - wox) / wcw)
        blocked = True
        if iyw >= 0 and iyw < wny and ixw >= 0 and ixw < wnx:
            blocked = wall_walk[iyw, ixw] == 0
        if blocked:
            vxn = 0.0
            vyn = 0.0
            xn = x[i]
            yn = y[i]
        x[i] = xn
        y[i] = yn
        vx[i] = vxn
        vy[i] = vyn
        pressure[i] = pn[i] / (two_pi * r[i])

        # wyjście przez bramę: przekroczenie odcinka bramy od środka
        left = False
        for k in range(K):
            if e_open[k] == 1:
                sd = (xn - e_ax[k]) * e_nx[k] + (yn - e_ay[k]) * e_ny[k]
                if sd > 0.0 and sd < 2.0:
                    td = (xn - e_ax[k]) * e_ux[k] + (yn - e_ay[k]) * e_uy[k]
                    if td >= -0.2 and td <= e_len[k] + 0.2:
                        state[i] = 2
                        exit_k[i] = k
                        t_exit[i] = tnow
                        left = True
                        break
        if left:
            continue

        if inj > 0.5:
            if pressure[i] > pthr:
                press_t[i] += dt
            else:
                press_t[i] = 0.0
            if press_t[i] >= hold:
                state[i] = 3
                vx[i] = 0.0
                vy[i] = 0.0


# ------------------------------------------------------------------------- wariant: wektorowy NumPy
def _bilerp_np(a, x, y, ox, oy, cs):
    ny, nx = a.shape
    fx = (x - ox) / cs - 0.5
    fy = (y - oy) / cs - 0.5
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    tx = fx - i0
    ty = fy - j0
    ia = np.clip(i0, 0, nx - 1)
    ib = np.clip(i0 + 1, 0, nx - 1)
    ja = np.clip(j0, 0, ny - 1)
    jb = np.clip(j0 + 1, 0, ny - 1)
    v0 = a[ja, ia] * (1 - tx) + a[ja, ib] * tx
    v1 = a[jb, ia] * (1 - tx) + a[jb, ib] * tx
    return v0 * (1 - ty) + v1 * ty


def step_numpy(x, y, vx, vy, r, m, state, mode, v0eff, target, wx, wy, cx, cy, coh,
               press_t, pressure, exit_k, t_exit,
               wall_nbx, wall_nby, wall_walk, wox, woy, wcw, wnx, wny,
               nav_ex, nav_ey, nox, noy, ncs, nnx, nny,
               e_ax, e_ay, e_ux, e_uy, e_nx, e_ny, e_len, e_open,
               P, dt, tnow,
               head, nxt, gox, goy, gcs, gnx, gny, fx, fy, pn):
    n = len(x)
    tau, A, B, kk, kappa, wA, wB, vcap, cfree, pthr, hold, inj, gap = P[:13]
    K = len(e_ax)
    act = np.flatnonzero(state != 2)
    mov = np.flatnonzero((state == 0) | (state == 1))
    fx[:] = 0.0
    fy[:] = 0.0
    pn[:] = 0.0
    if len(mov) == 0:
        return

    # --- napęd
    ex = np.zeros(n)
    ey = np.zeros(n)
    sm = np.flatnonzero(state == 1)
    for k in range(K):
        sel = sm[target[sm] == k]
        if len(sel) == 0:
            continue
        ex[sel] = _bilerp_np(nav_ex[k], x[sel], y[sel], nox, noy, ncs)
        ey[sel] = _bilerp_np(nav_ey[k], x[sel], y[sel], nox, noy, ncs)
    nrm = np.hypot(ex, ey)
    good = nrm > 1e-6
    ex = np.where(good, ex / np.where(good, nrm, 1), 0.0)
    ey = np.where(good, ey / np.where(good, nrm, 1), 0.0)
    wmask = (state == 0) & (mode == 1)
    dxw = wx - x
    dyw = wy - y
    dd = np.hypot(dxw, dyw)
    wm = wmask & (dd > 0.3)
    ex = np.where(wm, dxw / np.where(dd > 0, dd, 1), ex)
    ey = np.where(wm, dyw / np.where(dd > 0, dd, 1), ey)
    fx[mov] = m[mov] * (v0eff[mov] * ex[mov] - vx[mov]) / tau
    fy[mov] = m[mov] * (v0eff[mov] * ey[mov] - vy[mov]) / tau

    # --- pary agentów
    pos = np.column_stack((x[act], y[act]))
    tree = cKDTree(pos)
    rmax = float(r[act].max()) if len(act) else 0.35
    pairs = tree.query_pairs(r=2 * rmax + gap, output_type="ndarray")
    if len(pairs):
        i = act[pairs[:, 0]]
        j = act[pairs[:, 1]]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.hypot(dx, dy)
        rij = r[i] + r[j]
        keep = d < rij + gap
        i, j, dx, dy, d, rij = i[keep], j[keep], dx[keep], dy[keep], d[keep], rij[keep]
        d = np.maximum(d, 1e-9)
        nxv = np.where(d > 1e-9, dx / d, 1.0)
        nyv = np.where(d > 1e-9, dy / d, 0.0)
        g = rij - d
        gp = np.maximum(g, 0.0)
        fn = A * np.exp(np.minimum(g / B, 10.0)) + kk * gp
        tx, ty = -nyv, nxv
        dvt = (vx[j] - vx[i]) * tx + (vy[j] - vy[i]) * ty
        ft = kappa * gp * dvt
        fxp = fn * nxv + ft * tx
        fyp = fn * nyv + ft * ty
        fx += np.bincount(i, weights=fxp, minlength=n) - np.bincount(j, weights=fxp, minlength=n)
        fy += np.bincount(i, weights=fyp, minlength=n) - np.bincount(j, weights=fyp, minlength=n)
        pn += np.bincount(i, weights=kk * gp, minlength=n) + np.bincount(j, weights=kk * gp, minlength=n)

    # --- ściany
    xm, ym = x[mov], y[mov]
    iyw = ((ym - woy) / wcw).astype(int)
    ixw = ((xm - wox) / wcw).astype(int)
    inside = (iyw >= 0) & (iyw < wny) & (ixw >= 0) & (ixw < wnx)
    iyc = np.clip(iyw, 0, wny - 1)
    ixc = np.clip(ixw, 0, wnx - 1)
    bx = wall_nbx[iyc, ixc].astype(float)
    by = wall_nby[iyc, ixc].astype(float)
    dxw_ = xm - bx
    dyw_ = ym - by
    dw = np.hypot(dxw_, dyw_)
    ds = dw - 0.5 * wcw
    rm = r[mov]
    sel = inside & (dw > 1e-9) & ((ds - rm) < gap)
    nxw = np.where(sel, dxw_ / np.where(dw > 0, dw, 1), 0.0)
    nyw = np.where(sel, dyw_ / np.where(dw > 0, dw, 1), 0.0)
    gw = rm - ds
    gpw = np.where(sel, np.maximum(gw, 0.0), 0.0)
    fnw = np.where(sel, wA * np.exp(np.minimum(gw / wB, 10.0)) + kk * gpw, 0.0)
    txw, tyw = -nyw, nxw
    vt = vx[mov] * txw + vy[mov] * tyw
    ftw = -kappa * gpw * vt
    fx[mov] += fnw * nxw + ftw * txw
    fy[mov] += fnw * nyw + ftw * tyw
    pn[mov] += kk * gpw

    # --- spójność grupy
    cm = coh[mov] > 0
    if cm.any():
        dxg = cx[mov] - xm
        dyg = cy[mov] - ym
        dg = np.hypot(dxg, dyg)
        act_c = cm & (dg > cfree)
        mag = np.where(act_c, coh[mov] * np.minimum(dg - cfree, 3.0) * m[mov], 0.0)
        safe = np.where(dg > 0, dg, 1)
        fx[mov] += mag * dxg / safe
        fy[mov] += mag * dyg / safe

    # --- całkowanie
    vxn = vx[mov] + fx[mov] / m[mov] * dt
    vyn = vy[mov] + fy[mov] / m[mov] * dt
    sp = np.hypot(vxn, vyn)
    scale = np.where(sp > vcap, vcap / np.where(sp > 0, sp, 1), 1.0)
    vxn *= scale
    vyn *= scale
    xn = xm + vxn * dt
    yn = ym + vyn * dt
    iyw = ((yn - woy) / wcw).astype(int)
    ixw = ((xn - wox) / wcw).astype(int)
    inside = (iyw >= 0) & (iyw < wny) & (ixw >= 0) & (ixw < wnx)
    iyc = np.clip(iyw, 0, wny - 1)
    ixc = np.clip(ixw, 0, wnx - 1)
    blocked = ~inside | (wall_walk[iyc, ixc] == 0)
    vxn = np.where(blocked, 0.0, vxn)
    vyn = np.where(blocked, 0.0, vyn)
    xn = np.where(blocked, xm, xn)
    yn = np.where(blocked, ym, yn)
    x[mov], y[mov], vx[mov], vy[mov] = xn, yn, vxn, vyn
    pressure[mov] = pn[mov] / (2 * math.pi * r[mov])

    # --- wyjścia
    left = np.zeros(len(mov), dtype=bool)
    for k in range(K):
        if e_open[k] != 1:
            continue
        sd = (xn - e_ax[k]) * e_nx[k] + (yn - e_ay[k]) * e_ny[k]
        td = (xn - e_ax[k]) * e_ux[k] + (yn - e_ay[k]) * e_uy[k]
        hit = (~left) & (sd > 0) & (sd < 2.0) & (td >= -0.2) & (td <= e_len[k] + 0.2)
        if hit.any():
            idx = mov[hit]
            state[idx] = 2
            exit_k[idx] = k
            t_exit[idx] = tnow
            left |= hit

    # --- urazy
    if inj > 0.5:
        rest = mov[~left]
        over = pressure[rest] > pthr
        press_t[rest] = np.where(over, press_t[rest] + dt, 0.0)
        fall = rest[press_t[rest] >= hold]
        if len(fall):
            state[fall] = 3
            vx[fall] = 0.0
            vy[fall] = 0.0


# -------------------------------------------------------------------------------------- opakowanie
class Physics:
    def __init__(self, scene, cfg, A, wf, nav, open_flags, backend: str | None = None):
        self.scene = scene
        self.A = A
        self.P = pack_params(cfg)
        self.dt = float(cfg.sim.dt_s)
        want = backend or cfg.engine.backend
        if want == "auto":
            want = "numba" if HAVE_NUMBA else "numpy"
        if want == "numba" and not HAVE_NUMBA:
            warnings.warn("Numba niedostępna - używam wariantu NumPy.")
            want = "numpy"
        self.backend = want
        self._fallback_done = False
        # lista komórek agentów
        self.gcs = 1.6
        xmin, ymin, xmax, ymax = scene.bounds
        self.gox, self.goy = xmin - 4.0, ymin - 4.0
        self.gnx = int(math.ceil((xmax - xmin + 8.0) / self.gcs)) + 1
        self.gny = int(math.ceil((ymax - ymin + 8.0) / self.gcs)) + 1
        self.head = np.full(self.gnx * self.gny, -1, dtype=np.int32)
        self.nxt = np.full(A.n, -1, dtype=np.int32)
        self.fx = np.zeros(A.n)
        self.fy = np.zeros(A.n)
        self.pn = np.zeros(A.n)
        K = len(scene.exits)
        self.e_ax = np.array([e.a[0] for e in scene.exits])
        self.e_ay = np.array([e.a[1] for e in scene.exits])
        self.e_ux = np.array([e.u[0] for e in scene.exits])
        self.e_uy = np.array([e.u[1] for e in scene.exits])
        self.e_nx = np.array([e.n[0] for e in scene.exits])
        self.e_ny = np.array([e.n[1] for e in scene.exits])
        self.e_len = np.array([e.length for e in scene.exits])
        self.e_open = np.zeros(K, dtype=np.int8)
        self.set_fields(wf, nav, open_flags)

    def set_fields(self, wf, nav, open_flags):
        self.wf = wf
        self.nav = nav
        self.e_open[:] = np.asarray(open_flags, dtype=np.int8)

    def _args(self, tnow):
        A, s, wf, nav = self.A, self.scene, self.wf, self.nav
        return (A.x, A.y, A.vx, A.vy, A.r, A.m, A.state, A.mode, A.v0eff, A.target, A.wx, A.wy,
                A.cx, A.cy, A.coh, A.press_t, A.pressure, A.exit_k, A.t_exit,
                wf.nbx, wf.nby, wf.walk, s.x0, s.y0, s.cw, s.nx, s.ny,
                nav.ex, nav.ey, nav.x0, nav.y0, nav.cs, nav.nx, nav.ny,
                self.e_ax, self.e_ay, self.e_ux, self.e_uy, self.e_nx, self.e_ny, self.e_len, self.e_open,
                self.P, self.dt, tnow,
                self.head, self.nxt, self.gox, self.goy, self.gcs, self.gnx, self.gny,
                self.fx, self.fy, self.pn)

    def step(self, tnow: float):
        args = self._args(tnow)
        if self.backend == "numpy":
            step_numpy(*args)
        elif self.backend in ("numba", "loops"):
            try:
                step_loop(*args)
            except Exception as exc:  # błąd kompilacji Numby -> zapas
                if self.backend == "numba" and not self._fallback_done:
                    warnings.warn(f"Numba nie skompilowała jądra ({type(exc).__name__}: {str(exc)[:300]}). "
                                  "Przechodzę na wariant NumPy.")
                    self._fallback_done = True
                    self.backend = "numpy"
                    step_numpy(*args)
                else:
                    raise
        else:
            raise ValueError(f"Nieznany backend: {self.backend}")
