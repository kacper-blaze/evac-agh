"""Wspólny przebieg odniesienia dla testu regresyjnego (stare ustawienia, stara mapa)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def legacy_run(root, backend="numpy", n=400, tmax=40.0, party=5.0, seed=42):
    sys.path.insert(0, root)
    from evac.config import load_config
    from evac.sim import Simulation
    cfg, _ = load_config(None, {"map": "mapa_ms_poprawiona.json", "seed": seed,
                                "agents": {"count": n, "party_phase_s": party}, "sim": {"t_max_s": tmax},
                                # nowa fizyka kontaktu wyłączona - test pilnuje, że stary tryb się nie zmienił
                                "sfm": {"contact_damping_ratio": 0.0, "pressure_overlap_cap_m": 1e9},
                                "engine": {"speed_cap_mps": 5.0},
                                "map_options": {"unstick": False, "sliver_clearance_m": 0.0}}, [])
    sim = Simulation(cfg, os.path.join(root, "scenarios", "mapa_ms_poprawiona.json"), backend=backend, record=False)
    A = sim.run().agents
    return {"x": A.x.copy(), "y": A.y.copy(), "t_exit": np.nan_to_num(A.t_exit, nan=-1.0), "p": A.p.copy(),
            "state": A.state.astype(float), "target": A.target.astype(float)}
