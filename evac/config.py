"""Konfiguracja scenariusza: wartości domyślne + nadpisanie z pliku YAML + nadpisanie z linii poleceń.

Wszystkie czasy zdarzeń (close_at_s, hazard.start_s) liczone są OD ALARMU.
Alarm następuje po `agents.party_phase_s` sekundach "imprezy" (0 = alarm od razu).
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "map": "mapa_ms_poprawiona.json",
    "seed": 42,
    "sim": {
        "dt_s": 0.01,            # krok całkowania (sztywne siły kontaktowe -> mały krok)
        "t_max_s": 600.0,        # maksymalny czas OD ALARMU
        "coarse_dt_s": 0.25,     # co ile liczymy "wolne" rzeczy: zachowanie, panika, wybór bramy
        "record_dt_s": 0.5,      # co ile zapisujemy klatkę do wideo
    },
    "engine": {
        "backend": "auto",       # auto | numba | numpy
        "speed_cap_mps": 5.0,    # bezpiecznik numeryczny
    },
    "grid": {
        "wall_cell_m": 0.2,      # siatka pomocnicza: odległość od ścian
        "nav_cell_m": 0.5,       # siatka pomocnicza: pole nawigacji
        "smoke_cell_m": 1.0,     # siatka pomocnicza: dym
        "exit_pocket_m": 1.2,    # "kieszeń" za bramą, przez którą agent wychodzi
    },
    "sfm": {                     # Helbing, Farkas, Vicsek (2000)
        "mass_kg": 80.0,
        "tau_s": 0.5,
        "A_N": 2000.0,
        "B_m": 0.08,
        "k": 1.2e5,
        "kappa": 2.4e5,
        "wall_A_N": 2000.0,
        "wall_B_m": 0.08,
    },
    "agents": {
        "count": 3000,
        "radius_m": [0.25, 0.35],
        "v0_mps": {"mean": 1.3, "sd": 0.2, "min": 0.7, "max": 1.8},   # prędkość ucieczki
        "wander_speed_mps": 0.6,         # prędkość spaceru na "imprezie"
        "group_share": 0.5,              # odsetek osób w grupach
        "group_size": [2, 5],
        "party_phase_s": 60.0,           # czas "imprezy" przed alarmem (0 = od razu alarm)
        "stress_resistance": {"mean": 1.0, "sd": 0.25},
        "aggression": {"mean": 0.3, "sd": 0.15},
        "min_wall_dist_m": 0.5,
    },
    "behaviour": {
        "react_delay_s": [0.0, 30.0],    # losowe opóźnienie ruszenia po alarmie
        "ignore_alarm_share": 0.20,      # ignorują ostrzeżenie; ruszają, gdy ruszy większość sąsiadów
        "panic_initial_share": 0.05,     # od alarmu w panice, zarażają innych
        "panic_delay_s": [0.0, 2.0],
        "neighbor_radius_m": 5.0,        # promień "sąsiadów" dla reguły większości
        "majority": 0.5,                 # ułamek sąsiadów w ruchu, po którym rusza ignorujący
        "ignore_fallback_s": 180.0,      # bezpiecznik: po tylu s od alarmu ruszają wszyscy (null = wyłączony)
        "v_max_mps": 3.0,                # prędkość pożądana przy panice p=1
        "contagion": {"radius_m": 3.0, "rate_per_s": 0.15, "saturation": 3.0, "decay_per_s": 0.01},
        "group_cohesion": {"gain": 0.15, "free_dist_m": 2.0},
    },
    "hazard": {
        "enabled": True,
        "origin_xy": [80.0, 219.0],      # na lewo od toi-toi; zmieniamy tutaj
        "start_s": 0.0,
        "source_label": "UEK",
        "diffusion_m2s": 1.0,
        "source_rate": 2.0,
        "decay_per_s": 0.002,
        "full_conc": 1.0,
        "slow_factor": 0.5,              # o ile spowalnia dym o pełnym stężeniu
        "panic_rate_per_s": 0.05,
    },
    "exits": {
        "closed": [],                    # np. [Z2]; brama zamknięta od początku
        "close_at_s": {},                # np. {Z2: 45}; zamknięcie w trakcie (s od alarmu)
        "j_ref": 1.3,                    # przepływ referencyjny [os./(m*s)] do oceny kolejki
        "queue_radius_m": 4.0,
        "replan_s": 2.0,
        "hysteresis": 0.85,              # zmiana bramy, gdy nowy koszt < 85% obecnego
        "jam_agents": 10,
        "jam_flow_frac": 0.3,
        "jam_penalty_m": 40.0,
        "wall_cost": 2.0,
        "walking_speed_mps": 1.3,        # do przeliczenia czasu kolejki na metry
    },
    "injury": {
        "enabled": True,
        "threshold_Npm": 1600.0,         # Helbing 2000: suma sił promieniowych / obwód
        "hold_s": 0.5,                   # założenie własne: jak długo trzeba przekraczać próg
        "recover_s": 20.0,               # założenie własne: po tylu s przewrócony wstaje (null = leży do końca)
    },
    "barriers": [],                      # [{id: B1, from: [x,y], to: [x,y], thickness_m: 0.3}]
    "output": {
        "dir": "results",
        "video": True,
        "fps": 10,
        "trajectory": True,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def load_config(path: str | None = None, overrides: dict | None = None, sets: list[str] | None = None):
    """Zwraca (cfg_namespace, cfg_dict)."""
    user = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    if overrides:
        cfg = _merge(cfg, overrides)
    for s in sets or []:
        key, _, val = s.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return _to_ns(cfg), cfg
