"""Porównanie jądra pętlowego (tryb czystego Pythona, bez JIT) z wariantem NumPy na małym N."""
import os, sys, copy
os.environ["EVAC_NO_JIT"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from evac.config import load_config
from evac.sim import Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "..", "scenarios", "mapa_ms_poprawiona.json")


def run(backend, n=400, steps=300):
    cfg, _ = load_config(None, {"agents": {"count": n, "party_phase_s": 0}, "sim": {"t_max_s": 5}}, [])
    sim = Simulation(cfg, MAP, backend=backend, record=False)
    sim._alarm(0.0)
    sim.A.tmove[:] = 0.0
    sim.A.typ[:] = 0
    sim._activate(0.0)
    sim._update_queue_penalty(0.0); sim._psychology(0.0, 0.25)
    for i in range(steps):
        sim.phys.step(i * sim.phys.dt)
    return sim.A


def test_equivalence():
    a, b = run("numpy"), run("loops")
    d = max(np.abs(a.x - b.x).max(), np.abs(a.y - b.y).max())
    print("max |dx| =", d, " ewakuowani:", (a.state == 2).sum(), (b.state == 2).sum())
    assert d < 1e-6, d


if __name__ == "__main__":
    test_equivalence(); print("OK")
