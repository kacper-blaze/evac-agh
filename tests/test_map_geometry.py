"""Regresje geometrii mapy: obiekty czwartkowe, ławki i tunele wyjściowe."""
import os
import sys

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evac.config import load_config
from evac.geometry import load_scene


HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "..", "scenarios", "mapa_ms_poprawiona.json")


def _walk_at(scene, fields, point):
    iy, ix = scene.cell_of(point[0], point[1])
    return bool(fields.walk[iy, ix])


def test_osm_obstacles_are_blocked():
    cfg, _ = load_config()
    scene = load_scene(MAP, cfg)
    fields = scene.wall_fields(scene.initial_open())

    assert len(scene.benches) == 96
    assert len(scene.fixed_obstacles) == 10
    assert not _walk_at(scene, fields, [21.0, 19.0])  # siłownia zewnętrzna
    assert not _walk_at(scene, fields, [50.5, 60.0])  # boisko do siatkówki
    assert not _walk_at(scene, fields, scene.benches[0]["xy"])


def test_exits_are_at_outer_tunnel_ends():
    cfg, _ = load_config()
    scene = load_scene(MAP, cfg)
    fields = scene.wall_fields(scene.initial_open())
    original_area = Path(scene.polygon)
    expected = {
        "W1": np.array([119.745, 137.785]),
        "W2": np.array([3.45, 130.28]),
        "W3": np.array([112.90, 106.685]),
    }

    assert len(scene.tunnels) == 3
    components, _ = ndimage.label(fields.walk, structure=np.ones((3, 3), dtype=np.uint8))
    main_y, main_x = scene.cell_of(60.0, 150.0)
    main_component = components[main_y, main_x]
    assert main_component != 0
    for exit_id, center in expected.items():
        exit_ = scene.exits[scene.exit_index[exit_id]]
        assert np.allclose(exit_.center, center, atol=0.01)
        assert not original_area.contains_point(exit_.center)
        assert _walk_at(scene, fields, exit_.center - 0.6 * exit_.n)
        assert _walk_at(scene, fields, exit_.center + 0.6 * exit_.n)
        iy, ix = scene.cell_of(*(exit_.center - 0.6 * exit_.n))
        assert components[iy, ix] == main_component


if __name__ == "__main__":
    test_osm_obstacles_are_blocked()
    test_exits_are_at_outer_tunnel_ends()
    print("OK")
