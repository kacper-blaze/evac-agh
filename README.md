# Ewakuacja Miasteczka Studenckiego AGH – Social Force Model

Symulacja ewakuacji tłumu (Helbing, Farkas, Vicsek 2000). Wynik: wideo MP4, tabele CSV, wykresy.
Stan po etapie 2 – szczegóły zmian, założenia i wyniki kontrolne: **`ETAP2_WDROZENIE.md`**.

## Szybki start
```
pip install -r requirements.txt

# etap 2: nowa mapa (8 bram, płot L1), wszystkie nowe moduły
python run.py scenarios/etap2.yaml                              # pożar od grilla, wariant normalny
python run.py scenarios/etap2.yaml --variant remont_czwartek    # zamknięte W2, W4, Z1, Z2
python run.py scenarios/etap2.yaml --event huk --seed 7         # inne zdarzenie i miejsce (pozar | huk | pakunek | burza)
python run.py scenarios/etap2.yaml --set stewards.enabled=true  # służby porządkowe przy bramach
python run.py scenarios/etap2.yaml --set behaviour.group_return.share=0.15   # wracanie po znajomych
python run.py scenarios/etap2.yaml --n 6000 --no-video          # sama statystyka

# seria: 4 warianty x liczebności x ziarna -> results/matrix.csv + matrix_summary.csv/.md (średnia ± 95% CI)
python run_matrix.py scenarios/etap2.yaml --n 3000 6000 10000 --workers 4 --video-first

# etap 1 (stara mapa, stare zachowanie) - działa bez zmian
python run.py scenarios/juwenalia.yaml
```
Wyniki w `results/`: `*_summary.csv`, `*_gates.csv`, `*_timeseries.csv`, `*_agents.csv`, `*_events.csv`,
`*_ruszenie_vs_odleglosc.csv`, `*_krzywa_ewakuacji.png`, `*_mapa_zatorow.png`, `*.mp4`.

## Co jest czym
| plik | rola |
|---|---|
| `evac/config.py` | wszystkie parametry i wartości domyślne; warianty (`variants:`), zestawy zdarzeń (`event_presets:`) |
| `evac/geometry.py` | mapa → siatka: ściany, płoty (linie), tunele i futryny bram, osiągalność, przejście w płocie |
| `evac/fields.py` | nawigacja (Dijkstra, z kosztem dymu/stref), dym: dyfuzja (stary) i adwekcja z wiatrem (nowy) |
| `evac/agents.py` | tworzenie agentów: grupy, typy (zwykli / ignorujący / panikujący / gapie), skupiska, znajomość bram |
| `evac/engine.py` | fizyka SFM: jądro pętlowe (Numba) i wariant NumPy |
| `evac/sim.py` | pętla główna |
| `evac/events.py` | zdarzenia: pożar, huk, pakunek, burza, blokada bramy, awaria nagłośnienia |
| `evac/alarm.py` | alarm etapowy (nieświadomy → zauważył → zdecydował → ucieka) i gapie |
| `evac/choice.py` | wybór wyjścia: softmax, znajomość bram, naśladowanie, lokalny zator |
| `evac/injury.py` | obrażenia stopniowane (dawka urazu) |
| `evac/stats.py`, `evac/render.py` | tabele, wykresy, mapa zatorów, wideo |
| `scripts/bottleneck.py` | test wąskiego gardła vs literatura |
| `scripts/sensitivity.py` | analiza wrażliwości (jeden parametr naraz) |
| `scripts/make_extras.py` | generuje `scenarios/mapa_dodatki.json` (tunele, ławki, przykładowe grille i drzewa) |

## Mapa
* `scenarios/mapa_ms.json` – mapa zespołu, **bez żadnych ręcznych zmian**; po aktualizacji wystarczy ją podmienić.
* `scenarios/mapa_dodatki.json` – to, czego w mapie zespołu nie ma: tunele W1–W3 (jako wektor wyciągnięcia bramy),
  ławki z OSM, **przykładowe** grille i drzewa (`"przykladowe": true`). Jeśli mapa zespołu sama zawiera daną sekcję
  (`tunele`, `lawki`, `grille`, `drzewa`), wygrywa mapa.
* Loader jest tolerancyjny: brak sekcji = pusta; przeszkoda to `obrys`, `linia` (+`grubosc_m`) albo `xy` (+`promien_m`);
  nieznana forma = ostrzeżenie, nie błąd.
* Liczba osób (`agents.count`) dotyczy części **aktywnej** – tam, skąd da się dojść do otwartej bramy.

## Zasada przełączników
Każdy nowy moduł ma przełącznik w YAML-u, a wartości domyślne w `evac/config.py` odtwarzają zachowanie sprzed etapu 2
(`alarm.mode: legacy`, `exits.choice.mode: legacy`, `injury.mode: legacy`, `hazard.model: legacy`, `events: []`, ...).
`scenarios/etap2.yaml` włącza wszystko naraz; pojedyncze moduły można wyłączać przez `--set`, np.
`--set alarm.mode=legacy --set behaviour.density_speed.enabled=false`.

## Testy
```
python tests/test_regression.py     # stare ustawienia + to samo ziarno = wyniki identyczne z kodem sprzed zmian
python tests/test_new_map.py        # płot szczelny, 4 warianty, nikt w odciętej części, zdarzenia, alarm etapowy
python tests/test_backends.py       # zgodność jądra pętlowego i NumPy
python tests/test_map_geometry.py   # stara mapa
```

## Backend fizyki
`engine.backend: auto` używa Numby, jeśli jest zainstalowana (kompilacja sprawdzona, Numba 0.68); awaryjnie NumPy.
Wymuszenie: `--backend numpy` albo `--backend numba`.

## Ważne zastrzeżenia
* Parametry oznaczone jako wstępne / założenie są **ilustracyjne i niekalibrowane**; wyniki ilościowe podawać z zastrzeżeniem.
* Progi dawki urazu są dopasowane do liczb z literatury przytoczonych z pamięci (patrz `ETAP2_WDROZENIE.md`) –
  do sprawdzenia w źródłach; model obrażeń nie jest walidowany medycznie.
* Grille i drzewa są rozmieszczone przykładowo.
