"""Statystyki: tabele CSV + wykres krzywej ewakuacji."""
from __future__ import annotations

import csv
import os

import numpy as np

from .agents import EVACUATED, FALLEN, INJ_LIGHT, INJ_SEVERE, T_GAWKER, T_IGNORER, T_NORMAL, T_PANICKER


def _pct(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(res) -> dict:
    A, cfg = res.agents, res.cfg
    n = A.n
    evac = A.state == EVACUATED
    fallen = ~np.isnan(A.t_fall)          # przewrócił się choć raz
    lying = A.state == FALLEN
    te = A.t_exit[evac]
    s = {
        "N": n,
        "seed": cfg.seed,
        "backend": res.backend,
        "zamkniete_bramy_start": ",".join(e for e, o in zip(res.exit_ids, _initial_open(res)) if not o) or "-",
        "czas_symulacji_od_alarmu_s": round(res.t_end, 1),
        "czas_obliczen_s": round(res.wall_time_s, 1),
        "ewakuowani": int(evac.sum()),
        "przewroceni_choc_raz": int(fallen.sum()),
        "lezacy_na_koncu": int(lying.sum()),
        "pozostali_w_srodku": int(n - evac.sum()),
        "odsetek_ewakuowanych_%": round(100 * evac.mean(), 1),
        "T50_s": round(_pct(te, 50), 1) if len(te) else "",
        "T90_s": round(_pct(te, 90), 1) if len(te) else "",
        "T100_s": round(float(te.max()), 1) if len(te) else "",
        "max_cisnienie_Npm": round(float(max((r["max_pressure_Npm"] for r in res.ts), default=0.0)), 0),
        "max_cisnienie_chwilowe_Npm": round(float(max((r.get("max_pressure_raw_Npm", 0.0) for r in res.ts), default=0.0)), 0),
        "max_gestosc_2m_os_m2": round(float(max((max(v for k, v in r.items() if k.startswith("dens2m_"))
                                              for r in res.ts), default=0.0)), 2),
    }
    types = [("zwykli", T_NORMAL), ("ignorujacy", T_IGNORER), ("panikujacy", T_PANICKER)]
    if (A.typ == T_GAWKER).any():
        types.append(("gapie", T_GAWKER))
    for name, code in types:
        m = A.typ == code
        e = m & evac
        s[f"{name}_n"] = int(m.sum())
        s[f"{name}_sredni_czas_wyjscia_s"] = round(float(np.mean(A.t_exit[e])), 1) if e.any() else ""
        st = A.t_start[m]
        st = st[~np.isnan(st)]
        s[f"{name}_sredni_czas_ruszenia_s"] = round(float(np.mean(st)), 1) if len(st) else ""
        s[f"{name}_przewroceni"] = int((m & fallen).sum())
    s.update(_extended(res, evac))
    return s


def _extended(res, evac) -> dict:
    """Statystyki modułów etapu 2 (teren aktywny, obrażenia, alarm etapowy, gapie, zmiany bram)."""
    A, cfg = res.agents, res.cfg
    ar = res.area or {}
    ev = res.events_obj
    prim = ev.primary if ev is not None else None
    severe = A.inj == INJ_SEVERE
    light = A.inj == INJ_LIGHT
    out = {
        "wariant": cfg.variant or "-",
        "zdarzenie": (prim["type"] if prim else "-"),
        "zdarzenie_xy": f"{prim['xy'][0]:.0f};{prim['xy'][1]:.0f}" if (prim and prim["xy"] is not None) else "",
        "pole_aktywne_m2": round(ar.get("aktywne_m2", float("nan")), 0),
        "pole_wylaczone_m2": round(ar.get("wylaczone_m2", float("nan")), 0),
        "gestosc_poczatkowa_os_m2": round(A.n / max(ar.get("aktywne_m2", float("nan")), 1.0), 3),
        "wyszli_cali": int((evac & (A.inj == 0)).sum()),
        "wyszli_ranni": int((evac & light).sum()),
        "nie_wyszli_ciezko_ranni": int((severe & ~evac).sum()),
        "nie_wyszli_pozostali": int((~evac & ~severe).sum()),
        "poszkodowani_razem": int((light | severe).sum()),
        "czas_ewakuacji_zdolnych_s": round(float(np.nanmax(A.t_exit)), 1) if evac.any() else "",
        "gapie_szczyt": int(max((r.get("gawkers", 0) for r in res.ts), default=0)),
        "zmienili_brame_%": round(100 * float((A.n_switch > 0).mean()), 1),
        "wracali_po_znajomych": int((~np.isnan(A.t_return)).sum()),
        "przekierowani_przez_sluzby": int(A.redirected.sum()),
    }
    tn = A.t_notice
    ok = ~np.isnan(tn)
    out["sredni_czas_zauwazenia_s"] = round(float(tn[ok].mean()), 1) if ok.any() else ""
    # etapowość: czy czas ruszenia rośnie z odległością od zdarzenia (korelacja rang Spearmana)
    m = ~np.isnan(A.t_start) & ~np.isnan(A.dist_event)
    if m.sum() > 10:
        from scipy.stats import spearmanr
        out["korelacja_ruszenie_odleglosc"] = round(float(spearmanr(A.dist_event[m], A.t_start[m])[0]), 2)
    else:
        out["korelacja_ruszenie_odleglosc"] = ""
    peak = 0.0
    for g in gate_table(res):
        peak = max(peak, float(g["szczyt_przeplywu_10s_os_m_s"] or 0.0))
    out["max_szczyt_przeplywu_os_m_s"] = round(peak, 2)
    return out


def start_vs_distance(res, step=20.0) -> list[dict]:
    """Czas zauważenia / ruszenia w pasach odległości od zdarzenia (kryterium: ewakuacja etapowa)."""
    A = res.agents
    d = A.dist_event
    if np.isnan(d).all():
        return []
    rows = []
    for lo in np.arange(0.0, np.nanmax(d) + step, step):
        m = (d >= lo) & (d < lo + step)
        if not m.any():
            continue
        tn, ts_ = A.t_notice[m], A.t_start[m]
        rows.append({"odleglosc_od_m": lo, "odleglosc_do_m": lo + step, "osob": int(m.sum()),
                     "sredni_czas_zauwazenia_s": round(float(np.nanmean(tn)), 1) if (~np.isnan(tn)).any() else "",
                     "sredni_czas_ruszenia_s": round(float(np.nanmean(ts_)), 1) if (~np.isnan(ts_)).any() else "",
                     "nie_ruszylo": int(np.isnan(ts_).sum())})
    return rows


def _initial_open(res):
    return res.scene.initial_open()


def gate_table(res) -> list[dict]:
    A = res.agents
    rows = []
    last = res.ts[-1] if res.ts else {}
    for k, e in enumerate(res.scene.exits):
        m = (A.exit_k == k) & (A.state == EVACUATED)
        te = np.sort(A.t_exit[m])
        flow_peak = 0.0
        if len(te) > 1:
            bins = np.arange(np.floor(te[0]), te[-1] + 10.0, 10.0)
            cnt, _ = np.histogram(te, bins=bins)
            flow_peak = cnt.max() / 10.0 / e.width
        rows.append({
            "brama": e.id, "szerokosc_m": e.width, "zamykana": int(e.closable),
            "otwarta_na_koncu": res.open_final[k] if res.open_final else "",
            "wyszlo": int(m.sum()),
            "udzial_%": round(100 * m.sum() / max(1, (A.state == EVACUATED).sum()), 1),
            "pierwsza_osoba_s": round(float(te[0]), 1) if len(te) else "",
            "ostatnia_osoba_s": round(float(te[-1]), 1) if len(te) else "",
            "sredni_przeplyw_os_m_s": round(len(te) / max(te[-1] - te[0], 1e-6) / e.width, 2) if len(te) > 1 else "",
            "szczyt_przeplywu_10s_os_m_s": round(flow_peak, 2),
            "max_gestosc_2m_os_m2": round(max((r.get(f"dens2m_{e.id}", 0.0) for r in res.ts), default=0.0), 2),
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_outputs(res, outdir: str, tag: str = "run") -> dict:
    os.makedirs(outdir, exist_ok=True)
    s = summarize(res)
    paths = {}
    paths["summary"] = os.path.join(outdir, f"{tag}_summary.csv")
    _write_csv(paths["summary"], [s])
    paths["gates"] = os.path.join(outdir, f"{tag}_gates.csv")
    _write_csv(paths["gates"], gate_table(res))
    paths["timeseries"] = os.path.join(outdir, f"{tag}_timeseries.csv")
    _write_csv(paths["timeseries"], res.ts)
    paths["events"] = os.path.join(outdir, f"{tag}_events.csv")
    _write_csv(paths["events"], [{"t_s": round(t, 2), "zdarzenie": e} for t, e in res.events])
    # agenci
    A = res.agents
    paths["agents"] = os.path.join(outdir, f"{tag}_agents.csv")
    _write_csv(paths["agents"], [
        {"id": i, "typ": int(A.typ[i]), "grupa": int(A.gid[i]), "t_start_s": A.t_start[i],
         "t_exit_s": A.t_exit[i], "t_fall_s": A.t_fall[i],
         "brama": res.exit_ids[A.exit_k[i]] if A.exit_k[i] >= 0 else "",
         "odl_od_zdarzenia_m": round(float(A.dist_event[i]), 1), "t_zauwazenia_s": A.t_notice[i],
         "uraz": int(A.inj[i]), "dawka_Nspm": round(float(A.dose[i]), 0), "zmiany_bramy": int(A.n_switch[i])}
        for i in range(A.n)])
    svd = start_vs_distance(res)
    if svd:
        paths["start_dist"] = os.path.join(outdir, f"{tag}_ruszenie_vs_odleglosc.csv")
        _write_csv(paths["start_dist"], svd)
    paths["plot"] = os.path.join(outdir, f"{tag}_krzywa_ewakuacji.png")
    plot_curve(res, paths["plot"])
    if res.heat and bool(res.cfg.output.heatmap):
        paths["heatmap"] = os.path.join(outdir, f"{tag}_mapa_zatorow.png")
        plot_heatmap(res, paths["heatmap"])
    return {"paths": paths, "summary": s}


def plot_curve(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = res.ts
    t = np.array([r["t_s"] for r in ts])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax = axs[0]
    ax.plot(t, [r["evacuated"] for r in ts], label="ewakuowani", lw=2)
    ax.plot(t, [r["moving"] for r in ts], label="w ruchu")
    ax.plot(t, [r["idle"] for r in ts], label="stoją")
    ax.plot(t, [r["fallen"] for r in ts], label="leżący", color="crimson")
    for key, lab, col in (("unaware", "nieświadomi", "gray"), ("gawkers", "gapie", "tab:purple"),
                          ("injured_light", "lekko ranni", "orange"), ("injured_severe", "ciężko ranni", "k")):
        v = [r.get(key, 0) for r in ts]
        if max(v, default=0) > 0:
            ax.plot(t, v, label=lab, color=col, lw=1.2, ls="--")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("czas od alarmu [s]")
    ax.set_ylabel("liczba osób")
    ax.legend()
    ax.set_title("Przebieg ewakuacji")
    ax = axs[1]
    for eid in res.exit_ids:
        ax.plot(t, [r[f"out_{eid}"] for r in ts], label=eid)
    ax.set_xlabel("czas od alarmu [s]")
    ax.set_ylabel("wyszło przez bramę")
    ax.legend()
    ax.set_title("Bramy (narastająco)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_map(ax, sc, cfg=None, light=True):
    """Tło mapy wspólne dla wykresów i wideo: obrys, tunele, przeszkody, płot, ławki, drzewa, grille, podkład."""
    from matplotlib.patches import Polygon
    und = getattr(cfg.output, "underlay", None) if cfg is not None else None
    if und is not None and getattr(und, "path", None) and os.path.isfile(str(und.path)):
        import matplotlib.pyplot as plt
        ax.imshow(plt.imread(str(und.path)), extent=list(und.extent), alpha=float(getattr(und, "alpha", 0.6)),
                  zorder=0, origin="upper")
        fc = "none"
    else:
        fc = "#f4f1ea"
    ax.add_patch(Polygon(sc.polygon, closed=True, fc=fc, ec="#444", lw=1.2, zorder=0.5))
    for t in sc.tunnels:
        ax.add_patch(Polygon(t, closed=True, fc=fc, ec="#777", lw=0.8, zorder=0.5))
    for o in sc.fixed_obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="#8a8f98", ec="#444", lw=0.6, zorder=1))
    for o in sc.bench_obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="#9b6b43", ec="#5f3d25", lw=0.4, zorder=1))
    for o in sc.tree_obstacles:
        ax.add_patch(Polygon(o, closed=True, fc="#3f8f4a", ec="#2a6332", lw=0.4, zorder=1))
    for g in sc.grills:
        ax.plot([g["xy"][0]], [g["xy"][1]], marker="s", ms=5, color="#333", mec="#e8590c", mew=1.2, zorder=2)
    for b in sc.barriers:
        ax.plot([b["from"][0], b["to"][0]], [b["from"][1], b["to"][1]], color="k", lw=2, zorder=2)
    for ln in sc.lines:
        ax.plot(ln["pts"][:, 0], ln["pts"][:, 1], color="#222", lw=1.8, zorder=2)
    stg = getattr(sc, "steward_gates", None)
    if stg is not None:
        for k in np.flatnonzero(stg):
            c = sc.exits[k].center - 2.0 * sc.exits[k].n
            ax.plot([c[0]], [c[1]], marker="^", ms=7, color="#1971c2", mec="white", mew=0.6, zorder=6)
    if sc.fence_gate_xy is not None:
        ax.plot([sc.fence_gate_xy[0]], [sc.fence_gate_xy[1]], marker="D", ms=6, color="#f2c200", mec="k", zorder=3)
    x0, y0, x1, y1 = sc.bounds
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 5, x1 + 5)
    ax.set_ylim(y0 - 5, y1 + 5)


def plot_heatmap(res, path):
    """Gdzie powstają zatory: maks. gęstość i maks. ciśnienie w komórkach 2 x 2 m w całym przebiegu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, sc = res.heat, res.scene
    ny, nx = h["dens"].shape
    ext = [h["x0"], h["x0"] + nx * h["cell"], h["y0"], h["y0"] + ny * h["cell"]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 9), constrained_layout=True)
    for ax, key, lab, cmap in ((axs[0], "dens", "maks. gęstość [os/m²]", "magma_r"),
                               (axs[1], "press", "maks. ciśnienie [N/m]", "inferno_r")):
        draw_map(ax, sc)
        data = np.ma.masked_less_equal(h[key], 0)
        im = ax.imshow(data, origin="lower", extent=ext, cmap=cmap, alpha=0.85, zorder=3)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label=lab)
        for k, e in enumerate(sc.exits):
            ax.text(e.center[0] + 3 * e.n[0], e.center[1] + 3 * e.n[1], e.id, fontsize=8, weight="bold",
                    ha="center", va="center", color="#2e9e4f" if (not res.open_final or res.open_final[k]) else "#c92a2a")
        ax.set_title(lab)
    fig.savefig(path, dpi=120)
    plt.close(fig)
