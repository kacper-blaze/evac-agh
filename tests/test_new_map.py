"""Nowa mapa (mapa_ms.json): płot L1, 8 bram, 4 warianty, osiągalność, zdarzenia i moduły etapu 2."""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evac.agents import EVACUATED
from evac.config import load_config
from evac.geometry import load_scene
from evac.sim import Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "..", "scenarios", "etap2.yaml")
MAP = os.path.join(HERE, "..", "scenarios", "mapa_ms.json")
NORTH, SOUTH = (60.0, 150.0), (30.0, 40.0)
VARIANTS = {"normalny": 8, "remont": 6, "czwartek": 6, "remont_czwartek": 4}


def _scene(variant=None, over=None):
    cfg, _ = load_config(SCEN, over, [], variant=variant)
    sc = load_scene(MAP, cfg)
    op = sc.initial_open()
    return cfg, sc, op, sc.wall_fields(op)


def test_fence_is_sealed():
    """Siatka 0,2 m vs płot 0,15 m: z północy nie da się przejść na południe (także po przekątnej)."""
    _, sc, op, wf = _scene()
    assert len(sc.lines) == 1 and len(sc.exits) == 8
    lab, _ = ndimage.label(wf.walk, structure=np.ones((3, 3), dtype=np.uint8))
    n, s = lab[sc.cell_of(*NORTH)], lab[sc.cell_of(*SOUTH)]
    assert n != 0 and s != 0 and n != s
    reach = sc.reachable(wf, op)
    assert reach[sc.cell_of(*NORTH)] and not reach[sc.cell_of(*SOUTH)]
    ar = sc.area_report(wf, op)
    assert 2000 < ar["wylaczone_m2"] < 3500 and ar["aktywne_m2"] > 11000


def test_fence_gate_opens_south():
    over = {"map_options": {"fence_gate": {"enabled": True, "xy": [26.0, 59.5], "width_m": 2.0, "open": True}}}
    _, sc, op, wf = _scene(over=over)
    assert sc.reachable(wf, op)[sc.cell_of(*SOUTH)]
    over["map_options"]["fence_gate"]["open"] = False
    _, sc, op, wf = _scene(over=over)
    assert not sc.reachable(wf, op)[sc.cell_of(*SOUTH)]


def test_variants_and_gates():
    for variant, n_open in VARIANTS.items():
        _, sc, op, wf = _scene(variant)
        assert sum(op) == n_open, (variant, op)
        lab, _ = ndimage.label(wf.walk, structure=np.ones((3, 3), dtype=np.uint8))
        main = lab[sc.cell_of(*NORTH)]
        for e, o in zip(sc.exits, op):
            inner = e.center - 0.6 * e.n
            assert bool(wf.walk[sc.cell_of(*inner)]), e.id           # korytarz / futryna jest drożna
            assert lab[sc.cell_of(*inner)] == main, e.id             # i połączona z częścią aktywną
            assert bool(wf.walk[sc.cell_of(*(e.center + 0.6 * e.n))]) == bool(o), e.id


def test_tolerant_loader():
    """Mapa bez sekcji opcjonalnych i z nieznaną przeszkodą nie wywraca loadera."""
    import json
    from evac.geometry import Scene
    cfg, _ = load_config()
    d = json.load(open(MAP, encoding="utf-8"))
    for key in ("chodniki", "zalozenia", "granica_latlon", "przejscia_zamykane"):
        d.pop(key, None)
    d["przeszkody"].append({"id": "X", "rodzaj": "coś nowego"})
    d["nowa_sekcja"] = [1, 2, 3]
    sc = Scene(d, cfg)
    assert len(sc.exits) == 6 and len(sc.warnings) == 1


def _run(event, variant="normalny", n=300, tmax=240, backend=None, sets=()):
    cfg, _ = load_config(SCEN, {"agents": {"count": n, "party_phase_s": 5}, "sim": {"t_max_s": tmax}, "seed": 5,
                                "output": {"heatmap": False}}, list(sets), variant=variant, event=event)
    sim = Simulation(cfg, MAP, backend=backend, record=False)
    return sim, sim.run()


def test_nobody_in_cut_off_part_and_all_modules_run():
    for event in ("pozar", "huk", "pakunek"):
        sim, res = _run(event, variant="remont_czwartek", tmax=420)
        A = res.agents
        assert sim.reach is not None
        left = A.state != EVACUATED
        iy, ix = sim.scene.cell_of(A.x[left], A.y[left])
        assert sim.reach[iy, ix].all(), event                 # nikt nie został w odciętej części
        assert (A.state == EVACUATED).mean() > 0.9, (event, (A.state == EVACUATED).mean())
        used = set(np.unique(A.exit_k[A.exit_k >= 0]).tolist())
        closed = {sim.scene.exit_index[g] for g in ("W2", "W4", "Z1", "Z2")}
        assert not (used & closed), event                     # nikt nie wyszedł zamkniętą bramą


def test_staged_alarm_start_grows_with_distance():
    from scipy.stats import spearmanr
    sim, res = _run("pozar", n=500)
    A = res.agents
    m = ~np.isnan(A.t_start)
    rho = spearmanr(A.dist_event[m], A.t_start[m])[0]
    assert rho > 0.4, rho
    near, far = A.dist_event < 40, A.dist_event > 100
    assert np.nanmean(A.t_start[near]) < np.nanmean(A.t_start[far])


def test_storm_prefers_covered_passages():
    sim, res = _run("burza", n=400)
    A = res.agents
    covered = [sim.scene.exit_index[g] for g in ("W1", "W2", "W3", "W4", "W5", "W6")]
    assert (A.state == EVACUATED).mean() > 0.9
    assert np.isin(A.exit_k[A.exit_k >= 0], covered).mean() > 0.8      # większość wybiera zadaszone przejścia (wyjścia W)


def test_stewards_and_group_return():
    sim, res = _run("pozar", n=500, sets=["stewards.enabled=true", "behaviour.group_return.share=0.3"])
    A = res.agents
    assert (A.state == EVACUATED).mean() > 0.9
    assert sim.steward_gates.all()
    assert (~np.isnan(A.t_return)).sum() > 0          # ktoś wrócił po znajomego
    assert not A.returning.any()                       # i nikt nie został w tym stanie
    assert A.knows[A.redirected].all()


def test_numpy_backend_runs_new_modules():
    sim, res = _run("pakunek", n=150, tmax=120, backend="numpy")
    assert res.backend == "numpy" and (res.agents.state == EVACUATED).mean() > 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("OK")
