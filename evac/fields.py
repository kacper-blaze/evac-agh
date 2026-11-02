"""Pola pomocnicze: nawigacja do bram (Dijkstra na siatce) i dym (dyfuzja na grubej siatce)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

BIG = 1.0e6


def _fill_nearest(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Komórki nieprawidłowe dostają wartość z najbliższej prawidłowej."""
    if valid.all() or not valid.any():
        return arr
    _, ind = ndimage.distance_transform_edt(~valid, return_indices=True)
    return arr[ind[0], ind[1]]


class Nav:
    """Pole nawigacji: dla każdej bramy odległość (z karą przy ścianach) i kierunek marszu."""

    def __init__(self, scene, wf, open_flags, cfg):
        self.cs = float(cfg.grid.nav_cell_m)
        self.x0, self.y0 = scene.x0, scene.y0
        self.nx = int(np.ceil(scene.nx * scene.cw / self.cs))
        self.ny = int(np.ceil(scene.ny * scene.cw / self.cs))
        K = len(scene.exits)
        self.K = K
        xc = self.x0 + self.cs * (np.arange(self.nx) + 0.5)
        yc = self.y0 + self.cs * (np.arange(self.ny) + 0.5)
        XC, YC = np.meshgrid(xc, yc)
        iy = np.clip(((YC - scene.y0) / scene.cw).astype(int), 0, scene.ny - 1)
        ix = np.clip(((XC - scene.x0) / scene.cw).astype(int), 0, scene.nx - 1)
        dn = wf.d[iy, ix]
        walk = (wf.walk[iy, ix] > 0) & (dn >= 0.25)
        self.walk = walk
        wall_cost = float(cfg.exits.wall_cost)
        cell_cost = 1.0 + wall_cost * np.clip((1.2 - dn) / 1.2, 0.0, 1.0)

        idx = -np.ones((self.ny, self.nx), dtype=np.int64)
        n_nodes = int(walk.sum())
        idx[walk] = np.arange(n_nodes)

        rows, cols, w = [], [], []
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]  # (dy, dx)
        for dy, dx in shifts:
            ys0 = slice(0, self.ny - dy)
            ys1 = slice(dy, self.ny)
            if dx >= 0:
                xs0, xs1 = slice(0, self.nx - dx), slice(dx, self.nx)
            else:
                xs0, xs1 = slice(-dx, self.nx), slice(0, self.nx + dx)
            a_ok = walk[ys0, xs0]
            b_ok = walk[ys1, xs1]
            ok = a_ok & b_ok
            if dy != 0 and dx != 0:
                # przekątna tylko, gdy oba sąsiednie pola ortogonalne są wolne (bez ścinania rogów)
                if dx > 0:
                    c1 = walk[ys0, xs1]  # (y, x+dx)
                    c2 = walk[ys1, xs0]  # (y+dy, x)
                else:
                    c1 = walk[ys0, xs1]
                    c2 = walk[ys1, xs0]
                ok &= c1 & c2
            length = self.cs * (np.hypot(dy, dx))
            wgt = length * 0.5 * (cell_cost[ys0, xs0] + cell_cost[ys1, xs1])
            rows.append(idx[ys0, xs0][ok])
            cols.append(idx[ys1, xs1][ok])
            w.append(wgt[ok])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        w = np.concatenate(w)
        graph = coo_matrix((w, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

        D = np.full((K, self.ny, self.nx), np.inf, dtype=np.float64)
        for k, e in enumerate(scene.exits):
            if not open_flags[k]:
                continue
            pm = scene.pocket_masks[k][iy, ix] & walk
            src = idx[pm]
            if len(src) == 0:
                # awaryjnie: komórki najbliżej środka bramy
                dist = np.hypot(XC - e.center[0], YC - e.center[1])
                dist = np.where(walk, dist, np.inf)
                src = idx[np.unravel_index(np.argmin(dist), dist.shape)][None]
            dk = dijkstra(graph, directed=False, indices=src, min_only=True)
            tmp = np.full((self.ny, self.nx), np.inf)
            tmp[walk] = dk
            D[k] = tmp

        # D do oceny kosztu: wypełniamy komórki zablokowane wartością najbliższej wolnej
        Dc = np.empty_like(D)
        EX = np.zeros((K, self.ny, self.nx), dtype=np.float32)
        EY = np.zeros((K, self.ny, self.nx), dtype=np.float32)
        h = self.cs
        for k in range(K):
            Dk = D[k]
            valid = np.isfinite(Dk)
            if not valid.any():
                Dc[k] = BIG
                continue
            Dfill = _fill_nearest(np.where(valid, Dk, 0.0), valid)
            Dc[k] = Dfill
            # gradient maskowany (różnice tylko między prawidłowymi komórkami)
            gx = np.zeros_like(Dk)
            gy = np.zeros_like(Dk)
            Dz = np.where(valid, Dk, 0.0)
            vx_p = np.zeros_like(valid); vx_m = np.zeros_like(valid)
            vx_p[:, :-1] = valid[:, 1:]; vx_m[:, 1:] = valid[:, :-1]
            Dxp = np.zeros_like(Dk); Dxm = np.zeros_like(Dk)
            Dxp[:, :-1] = Dz[:, 1:]; Dxm[:, 1:] = Dz[:, :-1]
            both = vx_p & vx_m & valid
            gx = np.where(both, (Dxp - Dxm) / (2 * h), gx)
            only_p = vx_p & ~vx_m & valid
            gx = np.where(only_p, (Dxp - Dz) / h, gx)
            only_m = vx_m & ~vx_p & valid
            gx = np.where(only_m, (Dz - Dxm) / h, gx)
            vy_p = np.zeros_like(valid); vy_m = np.zeros_like(valid)
            vy_p[:-1, :] = valid[1:, :]; vy_m[1:, :] = valid[:-1, :]
            Dyp = np.zeros_like(Dk); Dym = np.zeros_like(Dk)
            Dyp[:-1, :] = Dz[1:, :]; Dym[1:, :] = Dz[:-1, :]
            bothy = vy_p & vy_m & valid
            gy = np.where(bothy, (Dyp - Dym) / (2 * h), gy)
            only_py = vy_p & ~vy_m & valid
            gy = np.where(only_py, (Dyp - Dz) / h, gy)
            only_my = vy_m & ~vy_p & valid
            gy = np.where(only_my, (Dz - Dym) / h, gy)
            mag = np.hypot(gx, gy)
            with np.errstate(invalid="ignore", divide="ignore"):
                ex = np.where(mag > 1e-9, -gx / mag, 0.0)
                ey = np.where(mag > 1e-9, -gy / mag, 0.0)
            good = valid & (mag > 1e-9)
            EX[k] = _fill_nearest(ex, good).astype(np.float32)
            EY[k] = _fill_nearest(ey, good).astype(np.float32)
        self.D = np.where(np.isfinite(Dc), Dc, BIG).astype(np.float32)
        self.D_raw = D
        self.ex = np.ascontiguousarray(EX)
        self.ey = np.ascontiguousarray(EY)

    def cost_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Odległość do każdej bramy z punktów (x, y): tablica (K, n)."""
        ix = np.clip(((x - self.x0) / self.cs).astype(int), 0, self.nx - 1)
        iy = np.clip(((y - self.y0) / self.cs).astype(int), 0, self.ny - 1)
        return self.D[:, iy, ix].astype(np.float64)


class Smoke:
    """Dym jako pole stężenia na grubej siatce (dyfuzja). To jedyne miejsce, gdzie używamy siatki
    jako "modelu" - opisuje zagrożenie, a nie ruch agentów."""

    def __init__(self, scene, cfg):
        h = float(cfg.grid.smoke_cell_m)
        self.h = h
        self.x0, self.y0 = scene.x0, scene.y0
        self.nx = int(np.ceil(scene.nx * scene.cw / h))
        self.ny = int(np.ceil(scene.ny * scene.cw / h))
        f = int(round(h / scene.cw))
        ny_f, nx_f = self.ny * f, self.nx * f
        pad = np.zeros((ny_f, nx_f))
        sub = scene.inside[: min(scene.ny, ny_f), : min(scene.nx, nx_f)]
        pad[: sub.shape[0], : sub.shape[1]] = sub
        self.mask = (pad.reshape(self.ny, f, self.nx, f).mean(axis=(1, 3)) > 0.5).astype(np.float64)
        hz = cfg.hazard
        self.D = float(hz.diffusion_m2s)
        self.rate = float(hz.source_rate)
        self.decay = float(hz.decay_per_s)
        self.full = float(hz.full_conc)
        ox, oy = hz.origin_xy
        self.src = (int((oy - self.y0) / h), int((ox - self.x0) / h))
        self.c = np.zeros((self.ny, self.nx))
        self.substeps_dt = 0.2 * h * h / max(self.D, 1e-9)  # stabilność: D*dt/h^2 <= 0.25

    def step(self, dt: float, emit: bool = True):
        remaining = dt
        while remaining > 1e-12:
            s = min(remaining, self.substeps_dt)
            c, m = self.c, self.mask
            if emit:
                j, i = self.src
                c[max(0, j - 1): j + 2, max(0, i - 1): i + 2] += self.rate * s
            lap = np.zeros_like(c)
            fx = m[:, 1:] * m[:, :-1]
            dx = (c[:, 1:] - c[:, :-1]) * fx
            lap[:, :-1] += dx
            lap[:, 1:] -= dx
            fy = m[1:, :] * m[:-1, :]
            dy = (c[1:, :] - c[:-1, :]) * fy
            lap[:-1, :] += dy
            lap[1:, :] -= dy
            c += self.D * s / (self.h ** 2) * lap
            c *= (1.0 - self.decay * s)
            np.clip(c, 0.0, None, out=c)
            remaining -= s

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(((x - self.x0) / self.h).astype(int), 0, self.nx - 1)
        iy = np.clip(((y - self.y0) / self.h).astype(int), 0, self.ny - 1)
        return self.c[iy, ix] / self.full

    def snapshot_u8(self) -> np.ndarray:
        return (np.clip(self.c / self.full, 0, 1) * 255).astype(np.uint8)
