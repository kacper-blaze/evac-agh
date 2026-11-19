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


def _seg_dist(p, a, b) -> float:
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 <= 0 else min(1.0, max(0.0, float((p - a) @ d) / L2))
    return float(np.hypot(*(p - (a + t * d))))


def _ring_dist(p, ring) -> float:
    return min(_seg_dist(p, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))


def _disc(xy, r, n=8) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((float(xy[0]) + r * np.cos(a), float(xy[1]) + r * np.sin(a)))


def merge_extras(data: dict, extras: dict | None) -> dict:
    """Dokłada do mapy zespołu elementy z pliku dodatków (tunele, ławki, grille, drzewa).
    Jeśli mapa sama zawiera daną sekcję, wygrywa mapa - plik zespołu można podmieniać bez edycji.

    Mapa może zawierać te elementy na dwa sposoby: w osobnych sekcjach (`lawki`, `drzewa`, `tunele`)
    albo - jak mapa_ms.json z listopada 2026 - jako `przeszkody` z polem `rodzaj` ("ławka", "drzewo")
    oraz polem `korytarz` przy wejściach W1-W6. Oba przypadki są rozpoznawane, żeby nie dublować przeszkód."""
    out = dict(data)
    kinds = {p.get("rodzaj") for p in out.get("przeszkody", [])}
    # korytarze z pola "korytarz" przy wejściach -> tunele (jeśli mapa nie ma własnej sekcji "tunele")
    if "tunele" not in out:
        tun = [{"id": f"K_{w['id']}", "wyjscie": w["id"], "obrys": w["korytarz"]}
               for w in list(out.get("wyjscia", [])) + list(out.get("przejscia_zamykane", []))
               if w.get("korytarz")]
        if tun:
            out["tunele"] = tun
    if not extras:
        return out
    own = {"lawki": "ławka" in kinds, "drzewa": "drzewo" in kinds}      # mapa ma je już jako przeszkody
    for key in ("tunele", "lawki", "grille", "drzewa", "bariery"):
        if key not in out and not own.get(key) and key in extras:
            out[key] = extras[key]
    if extras.get("przeszkody"):
        out["przeszkody"] = list(out.get("przeszkody", [])) + list(extras["przeszkody"])
    return out


def _apply_map_flags(data: dict) -> dict:
    flags = data.get("flagi")
    if not flags:
        return data
    if not isinstance(flags, dict):
        raise ValueError("Pole 'flagi' mapy musi być słownikiem.")

    def value(group, key, fallback):
        group_flags = flags.get(group, {})
        if not isinstance(group_flags, dict):
            raise ValueError(f"Pole 'flagi.{group}' mapy musi być słownikiem.")
        flag = group_flags.get(key, fallback)
        if not isinstance(flag, bool):
            raise ValueError(f"Flaga 'flagi.{group}.{key}' musi mieć wartość true albo false.")
        return flag

    out = dict(data)
    for key in ("wyjscia", "przejscia_zamykane"):
        out[key] = [
            {**item, "otwarte": value("wyjscia", item["id"], item.get("otwarte", True))}
            for item in data.get(key, [])
        ]

    obstacles = []
    plot_flags = flags.get("ploty", {})
    if not isinstance(plot_flags, dict):
        raise ValueError("Pole 'flagi.ploty' mapy musi być słownikiem.")
    for item in data.get("przeszkody", []):
        active = value("ploty", item.get("id"), value(
            "przeszkody", item.get("rodzaj"), item.get("aktywna", True)
        ))
        obstacles.append({**item, "aktywna": active})
    out["przeszkody"] = obstacles

    for key, flag_key in (("lawki", "ławka"), ("drzewa", "drzewo"), ("grille", "grill")):
        if key in data:
            out[key] = [
                {**item, "aktywna": value("przeszkody", flag_key, item.get("aktywna", True))}
                for item in data[key]
            ]
    return out


class Scene:
    """Mapa + pola pomocnicze.

    Loader jest celowo tolerancyjny (mapa jest jeszcze zmieniana): brakujące sekcje są puste,
    przeszkoda może być wielokątem (`obrys`), linią (`linia` + `grubosc_m`, np. płot) albo punktem
    (`xy` + `promien_m`); nieznane formy są pomijane z ostrzeżeniem zamiast błędu.
    """

    def __init__(self, data: dict, cfg, extra_barriers: list | None = None):
        data = _apply_map_flags(data)
        self.cfg = cfg
        self.data = data
        self.warnings: list[str] = []
        mo = getattr(cfg, "map_options", None)
        self.name = data.get("nazwa", "mapa")
        self.polygon = np.asarray(data["granica"], dtype=float)

        # ---- przeszkody: wielokąty / linie (płoty) / punkty
        self.fixed_obstacles, self.lines, point_obst = [], [], []
        for o in data.get("przeszkody", []):
            if not o.get("aktywna", True):
                continue
            if "obrys" in o:
                self.fixed_obstacles.append(np.asarray(o["obrys"], dtype=float))
            elif "linia" in o:
                self.lines.append({"id": o.get("id", "L"), "rodzaj": o.get("rodzaj", "płot"),
                                   "pts": np.asarray(o["linia"], dtype=float),
                                   "thickness_m": float(o.get("grubosc_m", 0.2))})
            elif "xy" in o:
                point_obst.append(_disc(o["xy"], float(o.get("promien_m", 0.4))))
            else:
                self.warnings.append(f"przeszkoda {o.get('id', '?')}: brak 'obrys' / 'linia' / 'xy' - pominięta")
        use = lambda name: mo is None or bool(getattr(mo, name, True))   # noqa: E731
        self.benches = [b for b in data.get("lawki", []) if b.get("aktywna", True)] if use("benches") else []
        # OSM zapisuje większość ławek jako punkty bez orientacji i wymiarów. W modelu
        # używamy kierunkowo neutralnego, ośmiokątnego rzutu o zadanym promieniu.
        self.bench_obstacles = [_disc(b["xy"], float(b.get("promien_m", 0.45))) for b in self.benches]
        self.trees = [t for t in data.get("drzewa", []) if t.get("aktywna", True)] if use("trees") else []
        self.grills = [g for g in data.get("grille", []) if g.get("aktywna", True)] if use("grills") else []
        self.tree_obstacles = [_disc(t["xy"], float(t.get("promien_m", 0.3))) for t in self.trees]
        self.grill_obstacles = [_disc(g["xy"], float(g.get("promien_m", 0.5))) for g in self.grills]
        self.obstacles = (self.fixed_obstacles + self.bench_obstacles + point_obst
                          + self.tree_obstacles + self.grill_obstacles)
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
        if not self.exits:
            raise ValueError("Mapa nie zawiera żadnych wyjść ('wyjscia' / 'przejscia_zamykane').")

        self._build_tunnels(data, mo)

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
        width = float(e.get("szerokosc_m", np.hypot(*(b - a))))
        ex = Exit(id=str(e["id"]), a=a, b=b, width=width, closable=closable,
                  open0=bool(e.get("otwarte", True)), desc=e.get("opis", ""))
        return ex

    def _build_tunnels(self, data, mo):
        """Korytarze za bramami: `tunele` z mapy (obrys albo wektor wyciągnięcia odcinka bramy) oraz
        opcjonalne futryny (`gate_frame_depth_m`). Brama trafia na ZEWNĘTRZNY koniec swojego korytarza."""
        self.tunnels = []
        self.covered_exits = set()                   # bramy z przejściem przez budynek (schronienie przy burzy)
        owner = {}                                   # id bramy -> indeks tunelu
        poly_path = Path(self.polygon)
        if mo is None or getattr(mo, "use_tunnels", True):
            for t in data.get("tunele", []):
                if not t.get("aktywny", True):
                    continue
                eid = t.get("wyjscie")
                if "obrys" in t:
                    quad = np.asarray(t["obrys"], dtype=float)
                elif "wektor" in t and eid in self.exit_index:
                    e = self.exits[self.exit_index[eid]]
                    v = np.asarray(t["wektor"], dtype=float)
                    quad = np.array([e.a, e.b, e.b + v, e.a + v])
                else:
                    self.warnings.append(f"tunel {t.get('id', '?')}: brak 'obrys' / 'wektor' lub nieznana brama - pominięty")
                    continue
                self.tunnels.append(quad)
                if eid in self.exit_index:
                    owner[eid] = len(self.tunnels) - 1
                    cov = getattr(mo, "covered_exits", None) if mo is not None else None
                    if cov is None or eid in cov:        # zadaszone przejścia (schronienie przy burzy)
                        self.covered_exits.add(eid)
        depth = float(getattr(mo, "gate_frame_depth_m", 0.0)) if mo is not None else 0.0
        if depth > 0:
            for e in self.exits:
                if e.id in owner:
                    continue
                u = (e.b - e.a) / np.hypot(*(e.b - e.a))
                n = np.array([-u[1], u[0]])
                c = 0.5 * (e.a + e.b)
                if poly_path.contains_point(c + 0.4 * n, radius=1e-9):
                    n = -n
                self.tunnels.append(np.array([e.a, e.b, e.b + depth * n, e.a + depth * n]))
                owner[e.id] = len(self.tunnels) - 1
        for eid, ti in owner.items():
            e = self.exits[self.exit_index[eid]]
            quad = self.tunnels[ti]
            best, best_score = None, 0.3
            for i in range(len(quad)):
                p, q = quad[i], quad[(i + 1) % len(quad)]
                mid = 0.5 * (p + q)
                if poly_path.contains_point(mid, radius=1e-9):
                    continue
                score = _ring_dist(mid, self.polygon)
                if score > best_score:
                    best, best_score = (p, q), score
            if best is None:
                continue
            if np.hypot(*(0.5 * (best[0] + best[1]) - 0.5 * (e.a + e.b))) > 0.05:
                e.a, e.b = best[0].copy(), best[1].copy()
                L = float(np.hypot(*(e.b - e.a)))
                if abs(L - e.width) > 0.05 * e.width:
                    e.width = L

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
            # n ma wskazywać NA ZEWNĄTRZ obszaru dostępnego. Wyjścia z korytarzem leżą
            # na jego zewnętrznym końcu, a więc poza głównym wielokątem mapy.
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
        self.solid_mask = np.zeros_like(self.inside)      # tylko bryły (toi-toie itp.) - blokują dym
        for o in self.fixed_obstacles:
            self.solid_mask |= self._poly_mask(o)
        for b in self.barriers:
            r = max(b["thickness_m"] / 2.0, 0.6 * self.cw)
            self.obstacle_mask |= self._segment_mask(b["from"], b["to"], r)
        # płoty (linie): promień >= 0.75 komórki, żeby raster był szczelny także po przekątnej
        mo = getattr(self.cfg, "map_options", None)
        kmin = float(getattr(mo, "line_min_radius_cells", 0.75)) if mo is not None else 0.75
        ext = float(getattr(mo, "line_extend_m", 0.3)) if mo is not None else 0.3
        self.fence_mask = np.zeros_like(self.inside)
        for ln in self.lines:
            pts = ln["pts"].copy()
            if len(pts) >= 2 and ext > 0:
                for i, j in ((0, 1), (-1, -2)):
                    d = pts[i] - pts[j]
                    pts[i] = pts[i] + d / max(np.hypot(*d), 1e-9) * ext
            r = max(ln["thickness_m"] / 2.0, kmin * self.cw)
            for p, q in zip(pts[:-1], pts[1:]):
                self.fence_mask |= self._segment_mask(p, q, r)
        self.fence_gate_xy = None
        fg = getattr(mo, "fence_gate", None) if mo is not None else None
        if fg is not None and fg.enabled and self.lines:
            xy = np.asarray(fg.xy, dtype=float) if fg.xy is not None else self._fence_midpoint()
            self.fence_gate_xy = xy
            if fg.open:   # otwarte przejście = wycięcie w płocie
                X, Y = np.meshgrid(self.xs, self.ys)
                self.fence_mask &= ~(np.hypot(X - xy[0], Y - xy[1]) <= float(fg.width_m) / 2.0)
        self.obstacle_mask |= self.fence_mask
        # "kieszeń" za każdą bramą: prostokąt wystający na zewnątrz obrysu
        self.pocket_masks = []
        for e in self.exits:
            quad = np.array([
                e.a - 0.3 * e.n, e.b - 0.3 * e.n,
                e.b + self.pocket * e.n, e.a + self.pocket * e.n,
            ])
            self.pocket_masks.append(self._poly_mask(quad))

    def _fence_midpoint(self):
        pts = max(self.lines, key=lambda l: len(l["pts"]))["pts"]
        seg = np.hypot(*np.diff(pts, axis=0).T)
        half, acc = seg.sum() / 2.0, 0.0
        for k, L in enumerate(seg):
            if acc + L >= half:
                return pts[k] + (pts[k + 1] - pts[k]) * (half - acc) / L
            acc += L
        return pts[-1]

    # ------------------------------------------------------------- osiągalność
    def reachable(self, wf: WallFields, open_flags) -> np.ndarray:
        """Maska komórek, z których da się dojść do którejkolwiek OTWARTEJ bramy."""
        labels, _ = ndimage.label(wf.walk, structure=np.ones((3, 3), dtype=np.uint8))
        good = set()
        for k in range(len(self.exits)):
            if open_flags[k]:
                good.update(np.unique(labels[self.pocket_masks[k] & (wf.walk > 0)]).tolist())
        good.discard(0)
        return np.isin(labels, list(good)) if good else np.zeros_like(self.inside)

    def area_report(self, wf: WallFields, open_flags) -> dict:
        reach = self.reachable(wf, open_flags)
        cell = self.cw ** 2
        walk_in = self.inside & (wf.walk > 0)
        return {"obrys_m2": float(self.area()),
                "dostepne_m2": float(walk_in.sum() * cell),
                "aktywne_m2": float((walk_in & reach).sum() * cell),
                "wylaczone_m2": float((walk_in & ~reach).sum() * cell)}

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
        mo = getattr(self.cfg, "map_options", None)
        sl = float(getattr(mo, "sliver_clearance_m", 0.0)) if mo is not None else 0.0
        wc = float(getattr(mo, "wall_clearance_m", 0.0)) if mo is not None else 0.0
        if wc > 0:
            # środek ciała nie może zbliżyć się do ściany bardziej niż wc: w klinach i szczelinach siła ścian
            # przeskakiwała między dwiema najbliższymi komórkami (rzędu 100 kN) i osoba drgała w miejscu
            walk &= ndimage.distance_transform_edt(walk, sampling=self.cw) >= wc
        if sl > 0:
            # szczeliny węższe niż ciało: lokalne maksimum odległości od ściany < sl -> komórka jest ścianą
            d0 = ndimage.distance_transform_edt(walk, sampling=self.cw)
            win = 2 * int(math.ceil(0.7 / self.cw)) + 1
            walk &= ~(ndimage.maximum_filter(d0, size=win) < sl)
        d, ind = ndimage.distance_transform_edt(walk, sampling=self.cw, return_indices=True)
        nbx = self.xs[ind[1]].astype(np.float32)
        nby = self.ys[ind[0]].astype(np.float32)
        return WallFields(walk=walk.astype(np.uint8), nbx=nbx, nby=nby, d=d.astype(np.float32))

    def free_points(self, wf: WallFields, min_wall_dist: float, max_points: int = 400000,
                    rng: np.random.Generator | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        """Współrzędne komórek wnętrza obrysu z zapasem od ścian (do losowania pozycji)."""
        ok = self.inside & (wf.walk > 0) & (wf.d >= min_wall_dist)
        if mask is not None:
            ok &= mask
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
    import os
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    extras_name = getattr(cfg, "map_extras", None)
    if extras_name:
        cand = [extras_name, os.path.join(os.path.dirname(os.path.abspath(map_path)), extras_name)]
        path = next((c for c in cand if os.path.isfile(c)), None)
        if path is None:
            raise FileNotFoundError(f"Nie znaleziono pliku dodatków mapy: {extras_name}")
        with open(path, "r", encoding="utf-8") as f:
            data = merge_extras(data, json.load(f))
    return Scene(data, cfg, extra_barriers=extra_barriers)
