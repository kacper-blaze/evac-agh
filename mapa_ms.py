"""
mapa_ms.py – stała mapa obszaru symulacji (Miasteczko Studenckie AGH).

Wszystkie dane są w pliku mapa_ms.json (punkty w metrach). Nic nie jest pobierane z internetu.

  granica             – obrys obszaru (czerwony na Twoim rysunku), lista punktów [x, y]
  wyjscia             – W1–W6: wejścia/wyjścia (niebieskie); otwieranie: słownik OTWARTE niżej
  przejscia_zamykane  – Z1, Z2: przejścia, które bywają zamknięte (żółte), flaga "otwarte"
  chodniki            – C1, C2, ...: oś chodnika + szerokość (lub obrys placu)
  przeszkody          – toi-toie T1–T8, płot L1, ławki B1–B88, pnie drzew D1–D105
                        (włączanie/wyłączanie: słownik PRZESZKODY niżej)
  powierzchnie        – S1 siłownia, S2 boisko do siatkówki, S3 boisko do koszykówki – TYLKO WIZUALNIE
                        (nie są przeszkodami, można po nich chodzić)
  kontury             – linie tylko do rysunku (wschodnia ściana Promyka, lewa linia Z1)
  wyjścia W1–W6 są na końcu 7-metrowych korytarzy przez budynki (pole "korytarz")

Układ: x na wschód, y na północ, metry. 0,0 = lewy dolny róg (podany w EPSG:2180 w pliku).

Użycie:
    python mapa_ms.py                       # rysuje mapa_ms.png
    python mapa_ms.py --zamknij Z1 Z2       # rysunek z zamkniętymi przejściami

W symulacji:
    from mapa_ms import wczytaj
    m = wczytaj(zamkniete=["Z1"])     # Z1 zamknięte, reszta wg pliku
    m.obszar      # shapely Polygon – gdzie mogą chodzić ludzie (cały teren minus przeszkody)
    m.chodniki    # shapely (Multi)Polygon – powierzchnia chodników (np. do preferencji tras)
    m.wyjscia     # {"W1": LineString, ...} – tylko otwarte (W i Z)
    m.granica     # shapely Polygon – sam obrys

Wymagania: pip install shapely matplotlib
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
PLIK = HERE / "mapa_ms.json"

# ============================================================================
#  WEJŚCIA / WYJŚCIA – True = otwarte, False = zamknięte
#  (te flagi mają pierwszeństwo przed polem "otwarte" w mapa_ms.json)
# ============================================================================
OTWARTE = {
    "W1": True,   # przejście przez Bonus (północne)        5,4 m
    "W2": True,   # przejście przez Bratek                  5,4 m
    "W3": True,   # przejście przez Bonus (południowe)      5,4 m
    "W4": True,   # przy narożniku Bratka                   5,4 m
    "W5": True,   # ściana Zaścianka, północne              4,0 m
    "W6": True,   # ściana Zaścianka, południowe            4,0 m
    "Z1": True,   # przy Filutku (zamykane)                 1,2 m
    "Z2": True,   # między Promykiem a Bonusem (zamykane)   7,5 m
}

# ============================================================================
#  PRZESZKODY – True = stoją na mapie (blokują ruch), False = usunięte
#  (mają pierwszeństwo przed polem "aktywna" w mapa_ms.json)
# ============================================================================
PRZESZKODY = {
    "toi-toi": True,               # T1–T8, 1,2 × 1,2 m
    "płot": True,                  # L1
    "ławka": True,                 # B1–B88, 1,8 × 0,6 m
    "drzewo": True,                # D1–D105, pień o średnicy 0,15–0,7 m
}


def _aktywna(p):
    return PRZESZKODY.get(p.get("rodzaj"), p.get("aktywna", True))


def _przedluz(punkty, d):
    """Linia przedłużona o d metrów na obu końcach (płot szczelnie dochodzi do ściany)."""
    import math
    p = [list(map(float, q)) for q in punkty]
    for i, j in ((0, 1), (-1, -2)):
        dx, dy = p[i][0] - p[j][0], p[i][1] - p[j][1]
        L = math.hypot(dx, dy) or 1.0
        p[i] = [p[i][0] + dx / L * d, p[i][1] + dy / L * d]
    return LineString(p)


@dataclass
class Mapa:
    dane: dict
    granica: Polygon
    obszar: object         # Polygon (albo MultiPolygon, gdy płot dzieli teren)
    chodniki: object       # Polygon / MultiPolygon
    wyjscia: dict          # otwarte wyjścia: id -> LineString (na granicy)
    zamkniete: list        # id zamkniętych przejść
    przeszkody: list       # Polygony aktywnych przeszkód


def wczytaj(plik=PLIK, zamkniete=None, otwarte=None) -> Mapa:
    """Stan wejść bierze się ze słownika OTWARTE na początku pliku.
    zamkniete / otwarte – listy id (np. ["Z1"]) nadpisujące go na czas jednego wczytania."""
    d = json.loads(Path(plik).read_text(encoding="utf-8"))
    zamkniete, otwarte = set(zamkniete or []), set(otwarte or [])
    granica = Polygon(d["granica"])
    przeszkody = []
    for p in d["przeszkody"]:
        if not _aktywna(p):
            continue
        if "obrys" in p:
            przeszkody.append(Polygon(p["obrys"]))
        else:
            przeszkody.append(_przedluz(p["linia"], 0.3).buffer(p.get("grubosc_m", 0.2) / 2, cap_style="flat"))
    obszar = granica.difference(unary_union(przeszkody)) if przeszkody else granica
    if obszar.geom_type != "Polygon":
        czesci = sorted(obszar.geoms, key=lambda g: g.area, reverse=True)
        czesci = [g for g in czesci if g.area > 1.0]
        if len(czesci) > 1:
            print(f"Uwaga: płot dzieli obszar na {len(czesci)} części: "
                  + ", ".join(f"{g.area:.0f} m²" for g in czesci))
        obszar = MultiPolygon(czesci) if len(czesci) > 1 else czesci[0]
    ch = [LineString(c["os"]).buffer(c["szerokosc_m"] / 2, cap_style="flat", join_style="round")
          if "os" in c else Polygon(c["obrys"]) for c in d.get("chodniki", [])]
    chodniki = unary_union(ch).intersection(obszar) if ch else Polygon()
    wyj, zamk = {}, []
    for w in d["wyjscia"] + d["przejscia_zamykane"]:
        domyslnie = OTWARTE.get(w["id"], w.get("otwarte", True))
        jest_otwarte = (domyslnie or w["id"] in otwarte) and w["id"] not in zamkniete
        if jest_otwarte:
            wyj[w["id"]] = LineString(w["odcinek"])
        else:
            zamk.append(w["id"])
    return Mapa(d, granica, obszar, chodniki, wyj, zamk, przeszkody)


def rysuj(m: Mapa, plik=None, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 13))

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    def fill(g, color=None, ec="none", lw=0, zorder=1, **kw):
        """Wielokąt z dziurami (dziury przezroczyste)."""
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            verts, codes = [], []
            for ring in [p.exterior, *p.interiors]:
                c = list(ring.coords)
                verts += c
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor=ec, lw=lw, zorder=zorder))

    fill(m.obszar, color="#dfe8cf", lw=0, zorder=1)                 # teren (trawa itp.)
    fill(m.chodniki, color="#cfc6b8", lw=0, zorder=2)               # chodniki
    for c in m.dane.get("chodniki", []):
        if "os" in c and LineString(c["os"]).length > 12:
            p = LineString(c["os"]).interpolate(0.5, normalized=True)
            ax.text(p.x, p.y, f"{c['id']}\n{c['szerokosc_m']:.1f} m", fontsize=5, ha="center", va="center",
                    color="#5a4a35", zorder=6)
    c = np.array(m.granica.exterior.coords)
    ax.plot(c[:, 0], c[:, 1], color="#d7191c", lw=2.2, zorder=3)
    for k in m.dane.get("kontury", []):           # linie pomocnicze (poza obszarem symulacji)
        c = np.array(k["linia"]); ax.plot(c[:, 0], c[:, 1], color="#d7191c", lw=2.2, zorder=3)
        if "ciag_dalszy_kropki" in k:
            c = np.array(k["ciag_dalszy_kropki"])
            ax.plot(c[:, 0], c[:, 1], color="#d7191c", lw=2.2, ls=(0, (1, 2.5)), zorder=3)
    # powierzchnie tylko wizualne (boiska, siłownia) – pod przeszkodami, nad chodnikami
    for p in m.dane.get("powierzchnie", []):
        kol_o, kol_i, napis = {"boisko do siatkówki": ("#5fa889", None, "siatkówka"),
                               "boisko do koszykówki": ("#5fa889", None, "koszykówka"),
                               "siłownia zewnętrzna": ("#d9cdb8", None, "siłownia")}.get(p["rodzaj"], ("#ccc", None, p["id"]))
        g = Polygon(p["obrys"]).intersection(m.granica)
        fill(g, color=kol_o, ec="white", lw=0.5, zorder=2.5)
        if p.get("pole_gry"):
            fill(Polygon(p["pole_gry"]).intersection(g), color=kol_i, ec="white", lw=0.6, zorder=2.6)
        cc = g.representative_point() if not p.get("pole_gry") else Polygon(p["pole_gry"]).centroid
        ax.text(cc.x, cc.y, napis, ha="center", va="center", fontsize=7, weight="bold",
                color="white" if p["rodzaj"].startswith("boisko") else "#5a4a35", zorder=7)
    toi = [p for p in m.dane["przeszkody"] if p.get("rodzaj") == "toi-toi" and _aktywna(p)]
    for p in m.dane["przeszkody"]:
        if not _aktywna(p):
            continue                      # wyłączone przeszkody nie są rysowane
        if "linia" in p:
            c = np.array(p["linia"])
            ax.plot(c[:, 0], c[:, 1], color="#4f7a3a" if p["rodzaj"] == "żywopłot" else "#222",
                    lw=3 if p["rodzaj"] == "żywopłot" else 1.8, zorder=4)
        else:
            if p.get("rodzaj") == "drzewo":            # pień + delikatny zarys korony
                sx, sy = p["srodek"]
                ax.add_patch(plt.Circle((sx, sy), p["srednica_korony_m"] / 2, facecolor="#6f9a4a", alpha=0.18,
                                        edgecolor="none", zorder=2))
            kolor = {"toi-toi": "#1d6fd1", "ławka": "#8a5a2b", "drzewo": "#5b3714", "siłownia zewnętrzna": "#e08a2c",
                     "boisko do siatkówki": "#e8b04a", "boisko do koszykówki": "#d9774b"}.get(p.get("rodzaj"), "#8d979b")
            if p.get("rodzaj") in ("siłownia zewnętrzna", "boisko do siatkówki", "boisko do koszykówki"):
                cc = Polygon(p["obrys"]).representative_point()
                ax.text(cc.x, cc.y, {"siłownia zewnętrzna": "siłownia", "boisko do siatkówki": "siatkówka",
                         "boisko do koszykówki": "koszykówka"}[p["rodzaj"]],
                        ha="center", va="center", fontsize=7, weight="bold", color="#5a3500", zorder=7)
            fill(Polygon(p["obrys"]), color=kolor, ec=kolor if p.get("rodzaj") == "drzewo" else "white",
                 lw=0.8 if p.get("rodzaj") == "drzewo" else 0.4, zorder=5)
    if toi:
        cc = unary_union([Polygon(p["obrys"]) for p in toi]).centroid
        ax.annotate(f"toi-toi ×{len(toi)}", (cc.x, cc.y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, weight="bold", color="#1d6fd1", zorder=8)
    for w in m.dane["wyjscia"] + m.dane["przejscia_zamykane"]:
        c = np.array(w["odcinek"])
        otw = w["id"] in m.wyjscia
        kolor = "#1f4fd1" if w["id"].startswith("W") else "#f2c200"
        ax.plot(c[:, 0], c[:, 1], color=kolor, lw=7, zorder=4, solid_capstyle="butt",
                ls="-" if otw else (0, (1, 1)))
        p = LineString(w["odcinek"]).interpolate(0.5, normalized=True)
        ax.annotate(f"{w['id']} {w['szerokosc_m']:.1f} m" + ("" if otw else " (zamknięte)"), (p.x, p.y),
                    xytext=(8, 0), textcoords="offset points", fontsize=8, weight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=kolor, lw=0.8))
    x0, y0, x1, y1 = m.granica.bounds
    for k in m.dane.get("kontury", []):           # zakres rysunku obejmuje też linie pomocnicze
        for p in k["linia"] + k.get("ciag_dalszy_kropki", []):
            x0, y0, x1, y1 = min(x0, p[0]), min(y0, p[1]), max(x1, p[0]), max(y1, p[1])
    ax.set_xticks(np.arange(0, x1 + 10, 10)); ax.set_yticks(np.arange(0, y1 + 10, 10))
    ax.tick_params(labelsize=7); ax.grid(color="#999", alpha=0.25, lw=0.5)
    ax.set_xlim(x0 - 5, x1 + 15); ax.set_ylim(y0 - 5, y1 + 5); ax.set_aspect("equal")
    ax.set_xlabel("x [m] (wschód)"); ax.set_ylabel("y [m] (północ)")
    ax.legend(handles=[Patch(color="#dfe8cf", label=f"obszar ({m.obszar.area:,.0f} m²)".replace(",", " ")),
                       Patch(color="#cfc6b8", label=f"chodniki ({m.chodniki.area:,.0f} m²)".replace(",", " ")),
                       Patch(color="#1d6fd1", label="toi-toi 1,2 × 1,2 m"),
                       Patch(color="#8a5a2b", label="ławka 1,8 × 0,6 m"),
                       Line2D([], [], marker="o", ls="", color="#5b3714", markersize=5, label="pień drzewa (korona: jasna zieleń)"),
                       Patch(color="#5fa889", label="boiska (tylko wizualnie)"),
                       Patch(color="#d9cdb8", label="siłownia (tylko wizualnie)"),
                       Line2D([], [], color="#d7191c", lw=2.2, label="granica (ściana)"),
                       Line2D([], [], color="#1f4fd1", lw=7, label="wyjście W"),
                       Line2D([], [], color="#f2c200", lw=7, label="przejście zamykane Z"),
                       Line2D([], [], color="#222", lw=1.8, label="płot")],
              loc="upper left", fontsize=8)
    ax.set_title(m.dane["nazwa"], loc="left", fontsize=10)
    if own:
        fig.tight_layout()
        if plik:
            fig.savefig(plik, dpi=220, bbox_inches="tight")
    return ax


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zamknij", nargs="*", default=[], help="id przejść do zamknięcia, np. Z1 Z2")
    a = ap.parse_args()
    m = wczytaj(zamkniete=a.zamknij)
    print(f"Granica: {len(m.dane['granica'])} punktów, pole {m.granica.area:.0f} m², obwód {m.granica.length:.0f} m")
    print(f"Obszar do chodzenia: {m.obszar.area:.0f} m², w tym chodniki {m.chodniki.area:.0f} m²")
    print("Otwarte wyjścia:", ", ".join(f"{k} ({v.length:.1f} m)" for k, v in m.wyjscia.items()))
    print("Zamknięte:", ", ".join(m.zamkniete) or "brak")
    rysuj(m, HERE / "mapa_ms.png")
    print("Zapisano mapa_ms.png")