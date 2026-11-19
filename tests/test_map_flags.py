import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evac.config import load_config
from evac.geometry import Scene


def test_map_flags_override_individual_object_settings():
    cfg, _ = load_config()
    data = {
        "nazwa": "mapa-testowa",
        "granica": [[0, 0], [20, 0], [20, 20], [0, 20]],
        "wyjscia": [
            {"id": "W1", "odcinek": [[0, 5], [0, 15]], "otwarte": True},
        ],
        "przejscia_zamykane": [
            {"id": "Z1", "odcinek": [[20, 5], [20, 15]], "otwarte": False},
        ],
        "przeszkody": [
            {"id": "P1", "rodzaj": "płot dodatkowy", "linia": [[5, 5], [5, 15]]},
            {"id": "P2", "rodzaj": "płot dodatkowy", "linia": [[10, 5], [10, 15]]},
            {"id": "D1", "rodzaj": "drzewo", "xy": [15, 15], "promien_m": 0.4},
        ],
        "flagi": {
            "wyjscia": {"W1": False, "Z1": True},
            "ploty": {"P1": False, "P2": True},
            "przeszkody": {"drzewo": False},
        },
    }

    scene = Scene(data, cfg)

    assert scene.initial_open() == [False, True]
    assert [line["id"] for line in scene.lines] == ["P2"]
    assert not scene.obstacles
