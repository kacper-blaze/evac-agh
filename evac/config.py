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
    "map_extras": None,          # plik z dodatkami do mapy (tunele, ławki, grille, drzewa) - patrz scenarios/mapa_dodatki.json
    "seed": 42,
    "variant": None,             # nazwa wariantu z `variants:` (nakładka na ustawienia)
    "variants": {                # warianty mapy 2x2 (można dopisać własne w YAML)
        "normalny": {"exits": {"closed": []}},
        "remont": {"exits": {"closed": ["W2", "W4"]}},
        "czwartek": {"exits": {"closed": ["Z1", "Z2"]}},
        "remont_czwartek": {"exits": {"closed": ["W2", "W4", "Z1", "Z2"]}},
    },
    "event_presets": {},         # nazwane zestawy zdarzeń: {huk: [{type: bang, ...}]}; wybór: --event huk
    "map_options": {
        "reachability": True,            # agenci tylko tam, skąd da się dojść do otwartej bramy
        "use_tunnels": True,             # korytarze przejść przez budynek ("tunele" z mapy / dodatków)
        "gate_frame_depth_m": 0.0,       # futryna: korytarz o tej głębokości za bramą bez tunelu (0 = brak, jak dotąd)
        "line_min_radius_cells": 0.75,   # płot (linia) zajmuje co najmniej tyle komórek siatki w każdą stronę -> szczelny
        "covered_exits": ["W1", "W2", "W3", "W4", "W5", "W6"],   # które bramy mają zadaszone przejście (burza); null = wszystkie z korytarzem
        "unstick": True,                 # osoby w komórce niedostępnej są przesuwane do najbliższego wolnego miejsca
        "wall_clearance_m": 0.0,          # środek ciała co najmniej tyle od środka komórki ściany (ok. 0,2 m od powierzchni); 0 = jak dotąd
        "sliver_clearance_m": 0.35,      # komórki, w których nie zmieści się nawet najmniejsza osoba (szczeliny, ślepe kąty), stają się ścianą; 0 = wyłączone
        "line_extend_m": 0.3,            # płot przedłużony na końcach (szczelnie dochodzi do ściany)
        "benches": True, "trees": True, "grills": True,
        "fence_gate": {"enabled": False, "xy": None, "width_m": 2.0, "open": False},   # przejście w płocie
    },
    "sim": {
        "dt_s": 0.01,            # krok całkowania (sztywne siły kontaktowe -> mały krok)
        "t_max_s": 600.0,        # maksymalny czas OD ALARMU
        "coarse_dt_s": 0.25,     # co ile liczymy "wolne" rzeczy: zachowanie, panika, wybór bramy
        "record_dt_s": 0.5,      # co ile zapisujemy klatkę do wideo
    },
    "engine": {
        "backend": "auto",       # auto | numba | numpy
        "speed_cap_mps": 3.0,    # bezpiecznik numeryczny (najszybszy sprint w tłumie; 5 m/s dawało drgania przy ścianach)
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
        "contact_damping_ratio": 0.5,    # tłumienie normalne kontaktu jako ułamek krytycznego (0 = brak, jak w oryginale Helbinga)
        "pressure_overlap_cap_m": 0.05,  # ściśnięcie ciała powyżej 5 cm nie zwiększa liczonego ciśnienia (20% głębokości klatki)
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
        # skupiska startowe wokół grilli (i opcjonalnie ławek); share = 0 -> rozmieszczenie jak dotąd
        "clusters": {"share": 0.0, "sigma_m": 4.0, "benches_weight": 0.0},
    },
    "alarm": {
        "mode": "legacy",                # legacy = wszyscy słyszą alarm w t=0; staged = alarm etapowy
        "info_speed_mps": 3.0,           # prędkość fali informacji od zdarzenia (wstępne, brak danych)
        "wave_range_m": None,            # zasięg fali (null = bez ograniczenia)
        "global_delay_s": [60.0, 120.0], # komunikat globalny po tylu s (losowo z przedziału); null = brak komunikatu
        "decide_delay_s": [1.0, 8.0],    # zauważył -> zdecydował
        "prep_delay_s": [0.0, 6.0],      # zdecydował -> ucieka
        "word_of_mouth": {"radius_m": 4.0, "rate_per_s": 0.25},   # uciekający sąsiedzi "informują" nieświadomych
        "smoke_notice_conc": 0.15,       # dym o takim stężeniu sam zwraca uwagę
        "keep_wandering": True,          # nieświadomi dalej spacerują jak na imprezie
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
        # gapie: typ agenta; po zauważeniu zdarzenia idą na punkt obserwacji zamiast uciekać (wymaga alarm.mode=staged)
        "gawkers": {
            "share": 0.0,                # odsetek gapiów (do ustalenia przez zespół)
            "max_dist_m": 70.0,          # gapiem zostaje tylko ten, kto zauważył zdarzenie z takiej odległości
            "watch_dist_m": [8.0, 16.0], # pierścień obserwacji wokół zdarzenia
            "leave_smoke_conc": 0.2,     # odchodzą, gdy dotrze dym
            "leave_panic": 0.5,          # ...albo własna panika przekroczy próg
            "leave_frac_of_peak": 0.4,   # ...albo gapiów zostanie mniej niż ten ułamek szczytu
            "min_watch_s": 15.0, "max_watch_s": 150.0,
        },
        # prędkość pożądana maleje z lokalną gęstością (diagram fundamentalny, Weidmann 1993) -> realistyczny przepływ
        "density_speed": {"enabled": False, "radius_m": 1.0, "rho_max": 5.4, "gamma": 1.913, "min_factor": 0.1},
        # wracanie po znajomych: część osób z grup, zamiast uciekać, idzie najpierw (pod prąd) po kolegę,
        # który jeszcze stoi albo leży; share = 0 -> wyłączone
        "group_return": {"share": 0.0, "max_dist_m": 30.0, "give_up_s": 40.0},
    },
    # służby porządkowe przy bramach: informują o wszystkich bramach, przekierowują z zatłoczonej bramy
    # do mniej obciążonej i uspokajają (przekierowanie wymaga exits.choice.mode = softmax)
    "stewards": {"enabled": False, "gates": "all", "radius_m": 25.0, "redirect_prob": 0.5, "calm_per_s": 0.05},
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
        # --- model rozszerzony (model: advection); "legacy" = sama dyfuzja jak dotąd
        "model": "legacy",
        "wind": {"speed_mps": 0.8, "to_deg": 0.0, "random_dir": False, "gust_sd_mps": 0.0},  # to_deg: dokąd wieje (0 = wschód, 90 = północ)
        "growth_s": 60.0,                # źródło rośnie ~t^2 do pełnej mocy przez tyle s (0 = od razu pełna)
        "burnout_s": None,               # po tylu s źródło gaśnie (null = pali się do końca)
        "blocked_by_buildings": True,    # dym nie przenika przez ściany i toi-toie (płot przepuszcza)
        "route_cost": 6.0,               # mnożnik kosztu trasy przy pełnym stężeniu (omijanie); 0 = wyłączone
        "nogo_conc": 0.8,                # od tego stężenia strefa "nie do wejścia"
        "nogo_cost": 40.0,               # dodatkowy mnożnik kosztu w strefie nie do wejścia
        "repulsion_mps2": 1.5,           # przyspieszenie odpychające wzdłuż -grad(c) przy pełnym stężeniu; 0 = wyłączone
        "nav_update_s": 10.0,            # co ile s przeliczamy trasy z uwzględnieniem dymu / stref
    },
    "events": [],                        # lista zdarzeń, np. [{type: fire, t_s: 0, at: grill}]; pusta = stary `hazard`
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
        "choice": {
            "mode": "legacy",            # legacy = najtańsza brama (+próg p=0.6); softmax = model rozszerzony
            "temperature_m": 12.0,       # losowość pierwszego wyboru (softmax po kosztach w metrach)
            "knowledge": {"default": 1.0},        # odsetek osób znających bramę, np. {default: 0.95, W6: 0.15}
            "see_dist_m": 25.0,          # z tej odległości (po trasie) agent widzi bramę i jej kolejkę
            "herding": {"radius_m": 5.0, "weight_m": 15.0, "learn_frac": 0.3, "learn_prob": 0.5},
            "local_jam": {"slow_mps": 0.25, "patience_s": 10.0, "near_gate_m": 6.0, "penalty_m": 40.0, "memory_s": 60.0},
            "rationality_gamma": 1.5,    # waga informacji o kolejce = (1-p)^gamma (stopniowo, zamiast progu)
            "rationality_min": 0.15,     # ...ale nie mniej niż tyle: nawet spanikowany w końcu rezygnuje z zatkanej bramy
            "width_bonus_m": 0.0,        # szersza brama jest atrakcyjniejsza: koszt -= bonus * ln(szerokość / 1 m)
        },
    },
    "injury": {
        "enabled": True,
        "threshold_Npm": 1600.0,         # Helbing 2000: suma sił promieniowych / obwód
        "hold_s": 0.5,                   # założenie własne: jak długo trzeba przekraczać próg
        "recover_s": 20.0,               # założenie własne: po tylu s przewrócony wstaje (null = leży do końca)
        # --- model stopniowany (mode: dose); progi dawki to ZAŁOŻENIA WŁASNE do uzupełnienia z literatury
        "mode": "legacy",
        "dose_threshold_Npm": None,      # próg, od którego rośnie dawka (null = threshold_Npm)
        "dose_light_Nspm": 1500.0,       # dawka [N/m * s] = czas x przekroczenie progu ciśnienia -> lekko ranny
        "dose_severe_Nspm": 12000.0,      # -> ciężko ranny (leży, nie wstaje)
        "fall_dose_Nspm": 500.0,         # dawka za sam upadek
        "trample_dose_Nspm_per_s": 150.0,  # dawka na sekundę za każdą osobę stojącą na leżącym
        "getup_s": 6.0,                  # leżący (nie ciężko ranny) próbuje wstać po tylu s, jeśli nikt po nim nie depcze
        "getup_max_s": 20.0,             # po tylu s wstaje mimo tłoku (o ile nie jest ciężko ranny)
        "rib_fracture_Npm": 1600.0,      # lekki uraz: ok. 3 kN na klatkę / (2*pi*0,3 m) - 20% ściśnięcia, złamania żeber (Viano i King 2000); 0 = dawka dose_light
        "pressure_tau_s": 1.0,           # wygładzanie ciśnienia (stała czasowa EMA): uraz powodują obciążenia utrzymane, nie piki kroku całkowania
        "pressure_cap_Npm": 3700.0,      # ~6,9 kN / (2*pi*0,3 m): obciążenie złamań kości (Atkinson 1998 za Stillem); wyżej dawka nie rośnie szybciej
        "light_speed_factor": 0.55,      # lekko ranny idzie wolniej
    },
    "barriers": [],                      # [{id: B1, from: [x,y], to: [x,y], thickness_m: 0.3}]
    "output": {
        "dir": "results",
        "video": True,
        "fps": 10,
        "trajectory": True,
        "heatmap": False,                # mapa maks. gęstości i ciśnienia (PNG)
        "underlay": None,                # podkład pod wideo/wykresy: {path: orto.png, extent: [xmin, xmax, ymin, ymax], alpha: 0.6}
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


def as_dict(x) -> dict:
    """Namespace / dict / None -> dict (dla słowników o dowolnych kluczach, np. knowledge, close_at_s)."""
    if not x:
        return {}
    return dict(vars(x)) if not isinstance(x, dict) else dict(x)


def load_config(path: str | None = None, overrides: dict | None = None, sets: list[str] | None = None,
                variant: str | None = None, event: str | None = None):
    """Zwraca (cfg_namespace, cfg_dict).

    Kolejność nakładania: wartości domyślne < plik YAML < wariant (`variants:`) < overrides (linia poleceń) < --set.
    """
    user = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    variant = variant or (overrides or {}).get("variant") or cfg.get("variant")
    if variant:
        if variant not in cfg["variants"]:
            raise KeyError(f"Nieznany wariant '{variant}'. Dostępne: {', '.join(cfg['variants'])}")
        cfg = _merge(cfg, cfg["variants"][variant] or {})
        cfg["variant"] = variant
    if event:
        if event not in cfg["event_presets"]:
            raise KeyError(f"Nieznany zestaw zdarzeń '{event}'. Dostępne: {', '.join(cfg['event_presets']) or '-'}")
        cfg["events"] = copy.deepcopy(cfg["event_presets"][event])
        cfg["event_name"] = event
    cfg.setdefault("event_name", None)
    if overrides:
        cfg = _merge(cfg, overrides)
    for s in sets or []:
        key, _, val = s.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(val))
    return _to_ns(cfg), cfg
