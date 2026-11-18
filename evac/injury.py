"""Model obrażeń stopniowany (injury.mode = dose). Zastępuje regułę "przewrócony wstaje po 20 s".

Dawka urazu D [N/m * s] rośnie, gdy ciśnienie przekracza próg (1600 N/m, Helbing 2000):
    dD/dt = max(0, ciśnienie - próg)          dla stojących,
    + `fall_dose` za upadek, + `trample_dose` na sekundę za każdą osobę stojącą na leżącym.
Stany: bez obrażeń -> lekko ranny (D >= dose_light; idzie wolniej) -> ciężko ranny (D >= dose_severe; leży, nie wstaje).
Leżący, który nie jest ciężko ranny, wstaje po `getup_s`, gdy nikt po nim nie depcze (najpóźniej po `getup_max_s`).

Źródła progów: lekki uraz = ok. 3 kN na klatkę (20% ściśnięcia, złamania żeber; Viano i King 2000) = 1600 N/m;
ciężki = dawka dopasowana do dwóch szacunków Hopkinsa i in. 1993 (6227 N -> 15 s, 1112 N -> 4-6 min). Szacunki
Hopkinsa dotyczą statycznego ucisku zdrowego dorosłego; model nie jest walidowany medycznie.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .agents import AW_FLEEING, FALLEN, IDLE, INJ_LIGHT, INJ_SEVERE, MOVING


def step(sim, tau, dt):
    A, ic = sim.A, sim.cfg.injury
    st = A.state
    standing = (st == IDLE) | (st == MOVING)
    thr = float(ic.threshold_Npm if ic.dose_threshold_Npm is None else ic.dose_threshold_Npm)
    # ciśnienie wygładzone (EMA) i ograniczone od góry: uraz wynika z obciążenia utrzymanego (Hopkins 1993 -
    # 6,2 kN przez 15 s, 1,1 kN przez 4-6 min), a nie z chwilowego piku kontaktu sztywnych sprężyn
    tau_p = float(ic.pressure_tau_s)
    a = 1.0 if tau_p <= 0 else 1.0 - np.exp(-dt / tau_p)
    A.pressure_s += a * (np.where(st == 2, 0.0, A.pressure) - A.pressure_s)
    cap = float(ic.pressure_cap_Npm)
    ps = np.minimum(A.pressure_s, cap) if cap > 0 else A.pressure_s
    A.dose[standing] += np.maximum(ps[standing] - thr, 0.0) * dt
    down = st == FALLEN
    new_down = down & np.isnan(sim.t_down)
    sim.t_down[new_down] = tau
    A.t_fall[new_down & np.isnan(A.t_fall)] = tau
    A.dose[new_down] += float(ic.fall_dose_Nspm)
    lying = np.flatnonzero(down)
    contact = np.zeros(len(lying), dtype=int)
    if len(lying) and standing.any():
        si = np.flatnonzero(standing)
        tree = cKDTree(np.column_stack((A.x[si], A.y[si])))
        contact = np.asarray(tree.query_ball_point(np.column_stack((A.x[lying], A.y[lying])),
                                                   r=A.r[lying] + 0.3, return_length=True))
        A.dose[lying] += float(ic.trample_dose_Nspm_per_s) * contact * dt
    # --- poziomy urazu
    # lekki uraz = obciążenie złamań żeber utrzymane (wygładzone ciśnienie >= próg); ciężki = dawka zgniecenia (Hopkins)
    rib = float(ic.rib_fracture_Npm)
    light = (((A.pressure_s >= rib) & (st != 2)) if rib > 0 else (A.dose >= float(ic.dose_light_Nspm))) & (A.inj < INJ_LIGHT)
    A.inj[light] = INJ_LIGHT
    sim.t_inj[light & np.isnan(sim.t_inj)] = tau
    severe = (A.dose >= float(ic.dose_severe_Nspm)) & (A.inj < INJ_SEVERE) & (st != 2)
    A.inj[severe] = INJ_SEVERE
    coll = severe & standing                     # osunięcie się ciężko rannego stojącego
    A.state[coll] = FALLEN
    A.vx[coll] = 0.0
    A.vy[coll] = 0.0
    sim.t_down[coll] = tau
    A.t_fall[coll & np.isnan(A.t_fall)] = tau
    # --- wstawanie
    if len(lying):
        t_l = tau - sim.t_down[lying]
        up = lying[(A.inj[lying] < INJ_SEVERE)
                   & (((t_l >= float(ic.getup_s)) & (contact == 0)) | (t_l >= float(ic.getup_max_s)))]
        if len(up):
            A.state[up] = MOVING
            A.press_t[up] = 0.0
            sim.t_down[up] = np.nan
            A.aw[up] = AW_FLEEING
            A.gawking[up] = False
            A.returning[up] = False
            A.mode[up] = 0
            A.tmove[up] = np.minimum(A.tmove[up], tau)
            ns = up[np.isnan(A.t_start[up])]
            A.t_start[ns] = tau
            sim._replan(up[A.target[up] < 0], force=True, tau=tau)
