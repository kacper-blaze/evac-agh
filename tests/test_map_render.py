import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evac.config import load_config
from evac.geometry import load_scene
from evac.stats import draw_map

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(HERE, "..", "scenarios", "etap2.yaml")
MAP = os.path.join(HERE, "..", "scenarios", "mapa_ms.json")


def test_draw_map_uses_reference_map_layers():
    cfg, _ = load_config(SCENARIO)
    scene = load_scene(MAP, cfg)
    fig, ax = plt.subplots(figsize=(9, 13))
    try:
        draw_map(ax, scene, cfg)
        labels = [text.get_text() for text in ax.texts]

        assert any(label.startswith("C") and "\n" in label for label in labels)
        assert "siatkówka" in labels
        assert any(label.startswith("W1 ") for label in labels)
        assert ax.get_legend() is not None
        assert ax.get_xlabel() == "x [m] (wschód)"
    finally:
        plt.close(fig)


def test_video_map_background_leaves_gates_to_animation():
    cfg, _ = load_config(SCENARIO)
    scene = load_scene(MAP, cfg)
    fig, ax = plt.subplots(figsize=(9, 13))
    try:
        draw_map(ax, scene, cfg, draw_exits=False, legend=False)
        labels = [text.get_text() for text in ax.texts]

        assert not any(label.startswith("W1 ") for label in labels)
        assert ax.get_legend() is None
    finally:
        plt.close(fig)
