"""Geometria: wczytanie mapy (JSON w metrach) i pola pomocnicze na siatce.

Agenci poruszają się w przestrzeni CIĄGŁEJ. Siatka służy tylko do szybkiego
sprawdzania: "gdzie jest ściana" (siła od ścian) i "w którą stronę do bramy" (pole nawigacji).

Układ współrzędnych: metry, x na wschód, y na północ (jak w pliku JSON).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy import ndimage


@dataclass
class Exit:
    id: str
    a: np.ndarray            # początek odcinka bramy
    b: np.ndarray            # koniec odcinka bramy
    width: float
    closable: bool
    open0: bool              # czy otwarta na starcie (z pliku)
    center: np.ndarray = field(default=None)
    u: np.ndarray = field(default=None)      # wektor jednostkowy wzdłuż bramy
    n: np.ndarray = field(default=None)      # wektor jednostkowy na zewnątrz obrysu
    length: float = 0.0
    desc: str = ""


@dataclass
class WallFields:
    walk: np.ndarray    # uint8 (ny, nx): 1 = można stać
    nbx: np.ndarray     # float32: x najbliższej "ściany" (środek komórki)
    nby: np.ndarray
    d: np.ndarray       # float32: odległość środka komórki do najbliższej ściany


class Scene:
    """Mapa + pola pomocnicze."""

    def __init__(self, data: dict, cfg, extra_barriers: list | None = None):
        self.cfg = cfg
        self.data = data
        self.name = data.get("nazwa", "mapa")
        self.polygon = np.asarray(data["granica"], dtype=float)
        self.tunnels = [np.asarray(t["obrys"], dtype=float) for t in data.get("tunele", [])
                        if t.get("aktywny", True)]
        self.fixed_obstacles = [np.asarray(o["obrys"], dtype=float) for o in data.get("przeszkody", [])
                                if o.get("aktywna", True)]
        self.benches = [b for b in data.get("lawki", []) if b.get("aktywna", True)]
        # OSM zapisuje większość ławek jako punkty bez orientacji i wymiarów. W modelu
        # używamy kierunkowo neutralnego, ośmiokątnego rzutu o zadanym promieniu.
        self.bench_obstacles = []
        for b in self.benches:
            x, y = map(float, b["xy"])
            r = float(b.get("promien_m", 0.45))
            a = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
            self.bench_obstacles.append(np.column_stack((x + r * np.cos(a), y + r * np.sin(a))))
        self.obstacles = self.fixed_obstacles + self.bench_obstacles
        self.sidewalks = data.get("chodniki", [])

        # bariery: z pliku JSON ("bariery") + z konfiguracji
        self.barriers = []
        for b in list(data.get("bariery", [])) + list(extra_barriers or []):
            self.barriers.append({
                "id": b.get("id", "B"),
                "from": np.asarray(b.get("from", b.get("od")), dtype=float),
                "to": np.asarray(b.get("to", b.get("do")), dtype=float),
                "thickness_m": float(b.get("thickness_m", b.get("grubosc_m", 0.3))),
            })

        self.exits: list[Exit] = []
        for e in data.get("wyjscia", []):
            self.exits.append(self._mk_exit(e, closable=False))
        for e in data.get("przejscia_zamykane", []):
            self.exits.append(self._mk_exit(e, closable=True))
        self.exit_index = {e.id: k for k, e in enumerate(self.exits)}

        extent_parts = [self.polygon] + self.tunnels
        extent_parts += [np.vstack((e.a, e.b)) for e in self.exits]
        extent = np.vstack(extent_parts)
        xmin, ymin = extent.min(axis=0)
        xmax, ymax = extent.max(axis=0)
        self.bounds = (float(xmin), float(ymin), float(xmax), float(ymax))

        g = cfg.grid
        self.cw = float(g.wall_cell_m)
        self.pocket = float(g.exit_pocket_m)
        margin = self.pocket + 2.0
        self.x0 = xmin - margin
        self.y0 = ymin - margin
        self.nx = int(math.ceil((xmax - xmin + 2 * margin) / self.cw))
        self.ny = int(math.ceil((ymax - ymin + 2 * margin) / self.cw))
        self.xs = self.x0 + self.cw * (np.arange(self.nx) + 0.5)
        self.ys = self.y0 + self.cw * (np.arange(self.ny) + 0.5)

        self._orient_exits()
        self._rasterize()

    # ------------------------------------------------------------------ bramy
    def _mk_exit(self, e: dict, closable: bool) -> Exit:
        seg = np.asarray(e["odcinek"], dtype=float)
        a, b = seg[0], seg[-1]
        ex = Exit(id=e["id"], a=a, b=b, width=float(e["szerokosc_m"]), closable=closable,
                  open0=bool(e.get("otwarte", True)), desc=e.get("opis", ""))
        return ex

    def _orient_exits(self):
        walkable_polys = [Path(self.polygon)] + [Path(t) for t in self.tunnels]

        def is_walkable(point):
            return any(poly.contains_point(point, radius=1e-9) for poly in walkable_polys)

        for e in self.exits:
            d = e.b - e.a
            e.length = float(np.hypot(*d))
            e.u = d / e.length
            e.center = 0.5 * (e.a + e.b)
            n = np.array([-e.u[1], e.u[0]])
            # n ma wskazywać NA ZEWNĄTRZ obszaru dostępnego. Wyjścia W1-W3 leżą
            # teraz na końcach tuneli, a więc poza głównym wielokątem mapy.
            if is_walkable(e.center + 0.4 * n):
                n = -n
            e.n = n

    # -------------------------------------------------------------- rasteryzacja
    def _centers(self, i0, i1, j0, j1):
        X, Y = np.meshgrid(self.xs[i0:i1], self.ys[j0:j1])
        return np.column_stack((X.ravel(), Y.ravel())), (j1 - j0, i1 - i0)

    def _poly_mask(self, poly: np.ndarray) -> np.ndarray:
        """Maska komórek, których środek leży w wielokącie (liczone tylko w jego bbox)."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        lo = poly.min(axis=0) - self.cw
        hi = poly.max(axis=0) + self.cw
        i0 = max(0, int((lo[0] - self.x0) / self.cw))
        i1 = min(self.nx, int((hi[0] - self.x0) / self.cw) + 2)
        j0 = max(0, int((lo[1] - self.y0) / self.cw))
        j1 = min(self.ny, int((hi[1] - self.y0) / self.cw) + 2)
        pts, shape = self._centers(i0, i1, j0, j1)
        inside = Path(poly).contains_points(pts).reshape(shape)
        mask[j0:j1, i0:i1] = inside
        return mask

    def _segment_mask(self, p: np.ndarray, q: np.ndarray, radius: float) -> np.ndarray:
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        lo = np.minimum(p, q) - radius - self.cw
        hi = np.maximum(p, q) + radius + self.cw
        i0 = max(0, int((lo[0] - self.x0) / self.cw))
        i1 = min(self.nx, int((hi[0] - self.x0) / self.cw) + 2)
        j0 = max(0, int((lo[1] - self.y0) / self.cw))
        j1 = min(self.ny, int((hi[1] - self.y0) / self.cw) + 2)
        pts, shape = self._centers(i0, i1, j0, j1)
        d = q - p
        L2 = float(d @ d)
        t = np.clip(((pts - p) @ d) / L2, 0, 1) if L2 > 0 else np.zeros(len(pts))
        proj = p + t[:, None] * d
        dist = np.hypot(*(pts - proj).T)
        mask[j0:j1, i0:i1] = (dist <= radius).reshape(shape)
        return mask

    def _rasterize(self):
        self.inside = self._poly_mask(self.polygon)
        self.tunnel_mask = np.zeros_like(self.inside)
        for t in self.tunnels:
            self.tunnel_mask |= self._poly_mask(t)
        self.obstacle_mask = np.zeros_like(self.inside)
        for o in self.obstacles:
            self.obstacle_mask |= self._poly_mask(o)
        for b in self.barriers:
            r = max(b["thickness_m"] / 2.0, 0.6 * self.cw)
            self.obstacle_mask |= self._segment_mask(b["from"], b["to"], r)
        # "kieszeń" za każdą bramą: prostokąt wystający na zewnątrz obrysu
        self.pocket_masks = []
        for e in self.exits:
            quad = np.array([
                e.a - 0.3 * e.n, e.b - 0.3 * e.n,
                e.b + self.pocket * e.n, e.a + self.pocket * e.n,
            ])
            self.pocket_masks.append(self._poly_mask(quad))

    def initial_open(self) -> list[bool]:
        """Które bramy są otwarte na starcie (plik + konfiguracja)."""
        closed = set(self.cfg.exits.closed or [])
        return [bool(e.open0) and e.id not in closed for e in self.exits]

    def wall_fields(self, open_flags) -> WallFields:
        walk = self.inside | self.tunnel_mask
        for k, e in enumerate(self.exits):
            if open_flags[k]:
                walk |= self.pocket_masks[k]
        walk &= ~self.obstacle_mask
        d, ind = ndimage.distance_transform_edt(walk, sampling=self.cw, return_indices=True)
        nbx = self.xs[ind[1]].astype(np.float32)
        nby = self.ys[ind[0]].astype(np.float32)
        return WallFields(walk=walk.astype(np.uint8), nbx=nbx, nby=nby, d=d.astype(np.float32))

    def free_points(self, wf: WallFields, min_wall_dist: float, max_points: int = 400000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
        """Współrzędne komórek wnętrza obrysu z zapasem od ścian (do losowania pozycji)."""
        ok = self.inside & (wf.walk > 0) & (wf.d >= min_wall_dist)
        jj, ii = np.nonzero(ok)
        pts = np.column_stack((self.xs[ii], self.ys[jj]))
        if len(pts) > max_points:
            rng = rng or np.random.default_rng(0)
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        return pts

    # ----------------------------------------------------------------- pomocnicze
    def cell_of(self, x, y):
        return ((np.asarray(y) - self.y0) / self.cw).astype(int), ((np.asarray(x) - self.x0) / self.cw).astype(int)

    def area(self) -> float:
        p = self.polygon
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def load_scene(map_path: str, cfg, extra_barriers: list | None = None) -> Scene:
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Scene(data, cfg, extra_barriers=extra_barriers)
