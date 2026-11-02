"""Statystyki: tabele CSV + wykres krzywej ewakuacji."""
from __future__ import annotations

import csv
import os

import numpy as np

from .agents import EVACUATED, FALLEN, T_IGNORER, T_NORMAL, T_PANICKER


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
        "max_gestosc_2m_os_m2": round(float(max((max(v for k, v in r.items() if k.startswith("dens2m_"))
                                              for r in res.ts), default=0.0)), 2),
    }
    for name, code in (("zwykli", T_NORMAL), ("ignorujacy", T_IGNORER), ("panikujacy", T_PANICKER)):
        m = A.typ == code
        e = m & evac
        s[f"{name}_n"] = int(m.sum())
        s[f"{name}_sredni_czas_wyjscia_s"] = round(float(np.mean(A.t_exit[e])), 1) if e.any() else ""
        st = A.t_start[m]
        st = st[~np.isnan(st)]
        s[f"{name}_sredni_czas_ruszenia_s"] = round(float(np.mean(st)), 1) if len(st) else ""
        s[f"{name}_przewroceni"] = int((m & fallen).sum())
    return s


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
         "brama": res.exit_ids[A.exit_k[i]] if A.exit_k[i] >= 0 else ""}
        for i in range(A.n)])
    paths["plot"] = os.path.join(outdir, f"{tag}_krzywa_ewakuacji.png")
    plot_curve(res, paths["plot"])
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
    ax.plot(t, [r["fallen"] for r in ts], label="przewróceni", color="crimson")
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
