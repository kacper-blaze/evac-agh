# Ewakuacja Miasteczka Studenckiego AGH – Social Force Model

Symulacja ewakuacji tłumu (Helbing, Farkas, Vicsek 2000) na mapie z `scenarios/mapa_ms_poprawiona.json`.
Wynik: wideo MP4 + tabele statystyk (CSV) + wykres krzywej ewakuacji.

## Szybki start
```
pip install -r requirements.txt
python run.py scenarios/juwenalia.yaml                       # 3000 osób, wszystkie bramy otwarte
python run.py scenarios/juwenalia.yaml --n 6000 --closed Z2   # 6000 osób, Z2 zamknięta
python run.py scenarios/juwenalia.yaml --no-video             # sama statystyka
python run_matrix.py scenarios/juwenalia.yaml --n 3000 6000 10000 --workers 4   # porównanie
```
Wyniki lądują w `results/`: `*_summary.csv`, `*_gates.csv`, `*_timeseries.csv`, `*_agents.csv`, `*_krzywa_ewakuacji.png`, `*.mp4`.

## Co jest czym
| plik | rola |
|---|---|
| `evac/config.py` | wszystkie parametry i wartości domyślne (nadpisywane w YAML lub `--set klucz=wartość`) |
| `evac/geometry.py` | mapa → siatka pomocnicza (ściany, zamknięte obszary, ławki, tunele i bramy) |
| `evac/fields.py` | nawigacja do bram (Dijkstra) i dym (dyfuzja) |
| `evac/agents.py` | tworzenie agentów, grupy, typy zachowania |
| `evac/engine.py` | fizyka SFM: jądro w pętlach (Numba `@njit`) i wariant NumPy |
| `evac/sim.py` | pętla główna: impreza → alarm → reakcje → panika → bramy |
| `evac/stats.py`, `evac/render.py` | tabele, wykres, wideo |

## Zachowanie agentów
* **zwykli** ruszają po losowym opóźnieniu 0–30 s od alarmu,
* **panikujący od początku** (5%) ruszają od razu, mają wysoką panikę i „zarażają” innych w promieniu 3 m,
* **ignorujący** (20%) nie reagują na alarm; ruszają, gdy w ruch ruszy **większość sąsiadów** (promień 5 m).
  Bezpiecznik `behaviour.ignore_fallback_s` (domyślnie 180 s, `null` = wyłączony) zapobiega czekaniu w nieskończoność.
* Wszyscy znają wszystkie bramy; wybierają według drogi + szacowanej kolejki. Spanikowani (p ≥ 0,6) patrzą tylko na odległość.
* Dym startuje w `hazard.origin_xy` (domyślnie [80, 219]), spowalnia i podnosi panikę.

## Backend fizyki
`engine.backend: auto` używa Numby, jeśli jest zainstalowana; jeśli kompilacja się nie uda, program sam przełącza się na NumPy
(i ostrzega). Wymuszenie: `--backend numpy` albo `--backend numba`.
Test zgodności obu jąder: `python tests/test_backends.py`.

## Ważne zastrzeżenia
* Parametry (panika, zarażanie, dym, czas reakcji, `injury.hold_s`, `injury.recover_s`) to **wartości ilustracyjne, niekalibrowane**.
* „Ciśnienie” liczone jest tylko z sił kontaktowych ciał (k·g) / obwód; próg urazu 1600 N/m z pracy Helbinga.
* Przewrócony agent wstaje po `injury.recover_s` (20 s – założenie własne; `null` = leży do końca i może zatkać bramę).
