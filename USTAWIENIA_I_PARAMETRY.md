# Uruchamianie symulacji i katalog parametrów

Dokument opisuje interfejsy uruchomieniowe projektu oraz wszystkie ustawienia
konfiguracyjne rozpoznawane przez symulator. Główne źródło wartości domyślnych
to `evac/config.py`; scenariusze YAML mogą je nadpisywać. Parametry modelu są
częściowo ilustracyjne i niekalibrowane — nie należy traktować ich jako
zwalidowanych prognoz.

## 1. Pojedyncza symulacja

Uruchomienie z katalogu `evac-agh`:

```powershell
python run.py [SCENARIUSZ] [opcje]
```

`SCENARIUSZ` to opcjonalna ścieżka do pliku YAML. Bez niej używane są wartości
domyślne z `evac/config.py`. Przykładowo:

```powershell
python run.py scenarios\etap2.yaml --variant remont_czwartek --event huk --seed 7
python run.py scenarios\etap2.yaml --n 6000 --party 0 --no-video
python run.py scenarios\etap2.yaml --set hazard.origin_xy=[100,200] --set behaviour.ignore_alarm_share=0.3
```

| Argument | Znaczenie |
|---|---|
| `-h`, `--help` | Pokaż pomoc i dostępne argumenty. |
| `SCENARIUSZ` | YAML scenariusza; opcjonalny. |
| `--n N` | Liczba agentów (`agents.count`). |
| `--seed LICZBA` | Ziarno losowe (`seed`). To samo ziarno odtwarza te same losowania przy tej samej konfiguracji i backendzie. |
| `--closed [BRAMA ...]` | Zamknij wskazane bramy od początku, np. `--closed Z1 Z2`. |
| `--variant NAZWA` | Nałóż wariant z `variants:` w YAML. Wbudowane nazwy: `normalny`, `remont`, `czwartek`, `remont_czwartek`; można dodać własne. |
| `--event NAZWA` | Wybierz zestaw zdarzeń z `event_presets:` w YAML; zastępuje listę `events:` scenariusza. |
| `--tmax SEKUNDY` | Maksymalny czas symulacji od alarmu (`sim.t_max_s`). |
| `--party SEKUNDY` | Długość fazy przed alarmem (`agents.party_phase_s`); `0` oznacza alarm od razu. |
| `--set KLUCZ=YAML` | Nadpisz ustawienie. Można powtarzać argument; obsługuje ścieżki kropkowe, np. `--set behaviour.gawkers.share=0.1`. |
| `--out KATALOG` | Katalog plików wynikowych; domyślnie `output.dir` (zwykle `results`). |
| `--tag TEKST` | Własny prefiks/nazwa plików wynikowych. |
| `--backend {auto,numba,numpy}` | Wybór backendu obliczeń fizyki. `auto` korzysta z Numby, jeśli jest dostępna; inaczej NumPy. |
| `--no-video` | Nie zapisuj ani nie renderuj wideo dla tego uruchomienia. |

Kolejność nadpisywania: **wartości domyślne → YAML → wariant → argumenty
bezpośrednie (`--n`, `--closed` itd.) → `--set`**. Wartość po `--set` jest
parsowana jako YAML, więc można podawać liczby, `true`/`false`, `null`, listy
i tekst. Przykłady:

```powershell
--set exits.closed=[Z1,Z2]
--set alarm.global_delay_s=null
--set map_options.fence_gate.enabled=true
--set map_options.fence_gate.xy=[80,219]
```

Jeśli trzeba ponownie otworzyć bramy zamknięte w YAML-u lub wariancie, użyj
`--set exits.closed=[]` (puste `--closed` nie tworzy nadpisania).

## 2. Seria przebiegów: `run_matrix.py`

```powershell
python run_matrix.py SCENARIUSZ [opcje]
```

| Argument | Znaczenie i wartość domyślna |
|---|---|
| `-h`, `--help` | Pokaż pomoc. |
| `SCENARIUSZ` | Wymagany YAML scenariusza. |
| `--n N [N ...]` | Jedna lub więcej liczebności; domyślnie `3000 6000 10000`. |
| `--variants NAZWA [NAZWA ...]` | Warianty mapy; domyślnie `normalny remont czwartek remont_czwartek`. |
| `--gates NAZWA [NAZWA ...]` | Tryb starszej konfiguracji bram zamiast wariantów: `wszystkie_otwarte`, `Z1_zamknieta`, `Z2_zamknieta`, `Z1_i_Z2_zamkniete`. |
| `--events NAZWA [NAZWA ...]` | Nazwy zestawów `event_presets`; domyślnie brak wymuszonego zestawu (`None`, czyli `events:` ze scenariusza). |
| `--seeds LICZBA [LICZBA ...]` | Jawna lista ziaren. Bez niej dobierana jest liczba ziaren wg N: 10 dla N ≤ 3000, 5 dla N ≤ 6000, 3 dla większego N; pierwsze ziarno to 1. |
| `--workers N` | Liczba procesów równoległych; domyślnie `1`. |
| `--out KATALOG` | Katalog wyników; domyślnie `results` w projekcie. |
| `--backend TEKST` | Backend przekazywany do symulatora; domyślnie wybór automatyczny. |
| `--video` | Renderuj wideo dla każdego przebiegu (wolne i zajmuje miejsce). |
| `--video-first` | Renderuj wideo tylko dla pierwszego ziarna każdej kombinacji. |
| `--set KLUCZ=YAML` | Nadpisanie konfiguracji; można podać wielokrotnie. |

Zapisuje `matrix.csv`, `matrix_summary.csv` oraz `matrix_summary.md`. Jeśli
podano jednocześnie `--video` i `--video-first`, `--video` i tak włącza wideo
dla każdego przebiegu. Opcje wariantów i `--gates` służą do dwóch odmiennych
trybów — przy `--gates` to on definiuje kombinacje bram.

## 3. Narzędzia pomocnicze

### Analiza wrażliwości

```powershell
python scripts\sensitivity.py SCENARIUSZ [opcje]
```

Argumenty: wymagany `SCENARIUSZ`; `--n` (domyślnie 1500), `--seeds LICZBA
[LICZBA ...]` (domyślnie `1 2 3`), `--variant`, `--only KLUCZ [KLUCZ ...]`
(ograniczenie do wybranych nazw z listy poniżej), `--backend`, `--out`
(domyślnie `results/sensitivity.csv`). Skrypt porównuje bazę z niską i wysoką
wartością każdego badanego parametru:

| Klucz | Niska | Wysoka |
|---|---:|---:|
| `behaviour.v_max_mps` | 1.6 | 2.4 |
| `alarm.info_speed_mps` | 1.5 | 6.0 |
| `alarm.wave_range_m` | 30 | 120 |
| `alarm.global_delay_s` | `[30,30]` | `[180,180]` |
| `behaviour.gawkers.share` | 0.0 | 0.15 |
| `behaviour.ignore_alarm_share` | 0.05 | 0.40 |
| `behaviour.contagion.rate_per_s` | 0.05 | 0.30 |
| `exits.choice.temperature_m` | 4 | 30 |
| `exits.choice.knowledge.W6` | 0.05 | 0.60 |
| `exits.choice.width_bonus_m` | 0 | 20 |
| `injury.dose_severe_Nspm` | 6000 | 24000 |
| `hazard.wind.speed_mps` | 0 | 2 |

### Test wąskiego gardła

```powershell
python scripts\bottleneck.py [opcje]
```

Argumenty: `--widths SZEROKOŚĆ [SZEROKOŚĆ ...]` (domyślnie `1.0 1.5 2.5
4.0 5.4` m), `--n` (250), `--seeds LICZBA [LICZBA ...]` (domyślnie `1 2`),
`--backend` (domyślnie automatyczny), `--csv PLIK` (opcjonalny zapis).

### Rysowanie mapy

```powershell
python mapa_ms.py [--zamknij [BRAMA ...]]
```

`--zamknij` przyjmuje identyfikatory bram, np. `--zamknij Z1 Z2`; skrypt
przygotowuje rysunek mapy, nie uruchamia symulacji.

## 4. Pełny katalog konfiguracji (wartości domyślne)

Każdy z poniższych kluczy można ustawić w YAML albo nadpisać przez `--set`.
Lista pokazuje `DEFAULTS` z `evac/config.py`; scenariusze
`scenarios/juwenalia.yaml` i `scenarios/etap2.yaml` celowo nadpisują część
wartości.

```yaml
map: mapa_ms_poprawiona.json
map_extras: null
seed: 42
variant: null
variants:
  normalny: {exits: {closed: []}}
  remont: {exits: {closed: [W2, W4]}}
  czwartek: {exits: {closed: [Z1, Z2]}}
  remont_czwartek: {exits: {closed: [W2, W4, Z1, Z2]}}
event_presets: {}

map_options:
  reachability: true
  use_tunnels: true
  gate_frame_depth_m: 0.0
  line_min_radius_cells: 0.75
  covered_exits: [W1, W2, W3, W4, W5, W6]
  unstick: true
  wall_clearance_m: 0.0
  sliver_clearance_m: 0.35
  line_extend_m: 0.3
  benches: true
  trees: true
  grills: true
  fence_gate:
    enabled: false
    xy: null
    width_m: 2.0
    open: false

sim:
  dt_s: 0.01
  t_max_s: 600.0
  coarse_dt_s: 0.25
  record_dt_s: 0.05
engine:
  backend: auto
  speed_cap_mps: 3.0
grid:
  wall_cell_m: 0.2
  nav_cell_m: 0.5
  smoke_cell_m: 1.0
  exit_pocket_m: 1.2

sfm:
  mass_kg: 80.0
  tau_s: 0.5
  A_N: 2000.0
  B_m: 0.08
  k: 120000.0
  kappa: 240000.0
  contact_damping_ratio: 0.5
  pressure_overlap_cap_m: 0.05
  wall_A_N: 2000.0
  wall_B_m: 0.08

agents:
  count: 3000
  radius_m: [0.25, 0.35]
  v0_mps: {mean: 1.3, sd: 0.2, min: 0.7, max: 1.8}
  wander_speed_mps: 0.6
  group_share: 0.5
  group_size: [2, 5]
  party_phase_s: 60.0
  stress_resistance: {mean: 1.0, sd: 0.25}
  aggression: {mean: 0.3, sd: 0.15}
  min_wall_dist_m: 0.5
  clusters: {share: 0.0, sigma_m: 4.0, benches_weight: 0.0}

alarm:
  mode: legacy
  info_speed_mps: 3.0
  wave_range_m: null
  global_delay_s: [60.0, 120.0]
  decide_delay_s: [1.0, 8.0]
  prep_delay_s: [0.0, 6.0]
  word_of_mouth: {radius_m: 4.0, rate_per_s: 0.25}
  smoke_notice_conc: 0.15
  keep_wandering: true

behaviour:
  react_delay_s: [0.0, 30.0]
  ignore_alarm_share: 0.20
  panic_initial_share: 0.05
  panic_delay_s: [0.0, 2.0]
  neighbor_radius_m: 5.0
  majority: 0.5
  ignore_fallback_s: 180.0
  v_max_mps: 3.0
  contagion: {radius_m: 3.0, rate_per_s: 0.15, saturation: 3.0, decay_per_s: 0.01}
  group_cohesion: {gain: 0.15, free_dist_m: 2.0}
  gawkers:
    share: 0.0
    max_dist_m: 70.0
    watch_dist_m: [8.0, 16.0]
    leave_smoke_conc: 0.2
    leave_panic: 0.5
    leave_frac_of_peak: 0.4
    min_watch_s: 15.0
    max_watch_s: 150.0
  density_speed: {enabled: false, radius_m: 1.0, rho_max: 5.4, gamma: 1.913, min_factor: 0.1}
  group_return: {share: 0.0, max_dist_m: 30.0, give_up_s: 40.0}

stewards:
  enabled: false
  gates: all
  radius_m: 25.0
  redirect_prob: 0.5
  calm_per_s: 0.05

hazard:
  enabled: true
  origin_xy: [80.0, 219.0]
  start_s: 0.0
  source_label: UEK
  diffusion_m2s: 1.0
  source_rate: 2.0
  decay_per_s: 0.002
  full_conc: 1.0
  slow_factor: 0.5
  panic_rate_per_s: 0.05
  model: legacy
  wind: {speed_mps: 0.8, to_deg: 0.0, random_dir: false, gust_sd_mps: 0.0}
  growth_s: 60.0
  burnout_s: null
  blocked_by_buildings: true
  route_cost: 6.0
  nogo_conc: 0.8
  nogo_cost: 40.0
  repulsion_mps2: 1.5
  nav_update_s: 10.0

events: []
exits:
  closed: []
  close_at_s: {}
  j_ref: 1.3
  queue_radius_m: 4.0
  replan_s: 2.0
  hysteresis: 0.85
  jam_agents: 10
  jam_flow_frac: 0.3
  jam_penalty_m: 40.0
  wall_cost: 2.0
  walking_speed_mps: 1.3
  choice:
    mode: legacy
    temperature_m: 12.0
    knowledge: {default: 1.0}
    see_dist_m: 25.0
    herding: {radius_m: 5.0, weight_m: 15.0, learn_frac: 0.3, learn_prob: 0.5}
    local_jam: {slow_mps: 0.25, patience_s: 10.0, near_gate_m: 6.0, penalty_m: 40.0, memory_s: 60.0}
    rationality_gamma: 1.5
    rationality_min: 0.15
    width_bonus_m: 0.0

injury:
  enabled: true
  threshold_Npm: 1600.0
  hold_s: 0.5
  recover_s: 20.0
  mode: legacy
  dose_threshold_Npm: null
  dose_light_Nspm: 1500.0
  dose_severe_Nspm: 12000.0
  fall_dose_Nspm: 500.0
  trample_dose_Nspm_per_s: 150.0
  getup_s: 6.0
  getup_max_s: 20.0
  rib_fracture_Npm: 1600.0
  pressure_tau_s: 1.0
  pressure_cap_Npm: 3700.0
  light_speed_factor: 0.55

barriers: []
output:
  dir: results
  video: true
  fps: 20
  render_progress_every: 100
  trajectory: true
  heatmap: false
  underlay: null
```

### Znaczenie i zależności parametrów

- `map` wskazuje plik mapy JSON; `map_extras` opcjonalny plik uzupełniający.
  Ścieżki są szukane względem bieżącego katalogu, katalogu YAML/mapy oraz
  katalogu `scenarios` (zależnie od pliku).
- `sim.dt_s` steruje krokiem fizyki, `coarse_dt_s` krokiem zachowania i
  aktualizacji wolniejszych procesów, `record_dt_s` odstępem zapisu klatek,
  a `t_max_s` maksymalnym czasem od alarmu. Domyślne `record_dt_s: 0.05` i
  `output.fps: 20` dają odtwarzanie w czasie rzeczywistym (jedna klatka co
  0,05 s symulacji); by zachować tę prędkość po zmianie FPS, ustaw
  `record_dt_s` na `1 / fps`.
- `engine.backend` i `--backend` wybierają `auto`, `numba` lub `numpy`.
  `speed_cap_mps` ogranicza prędkość w jądrze fizyki.
- `grid.*_cell_m` ustawia rozdzielczość siatek ścian, nawigacji i dymu;
  zmniejszenie komórki zwiększa koszt obliczeń i pamięci.
- `sfm.*` są parametrami modelu Social Force: masa, czas relaksacji,
  oddziaływania społeczne, sprężystość/tarcie kontaktu i siły ścian.
- `agents.*` określa populację i rozkłady cech. `v0_mps` ma postać
  `{mean, sd, min, max}`, `radius_m` i `group_size` są przedziałami włącznie.
  `clusters` steruje udziałem grup rozpoczynających blisko grilli/ławek.
- `alarm.mode`: `legacy` powoduje alarm jednocześnie dla wszystkich;
  `staged` rozsyła informację falą, sąsiadami, dymem i/lub komunikatem.
  Opóźnienia i zasięgi `alarm.*` wyrażane są w sekundach/metrach, chyba że
  nazwa wskazuje inne jednostki.
- `behaviour.*` ustawia typy agentów, opóźnienia reakcji, panikę, zarażanie,
  spójność/odnajdywanie grup. `density_speed.enabled` włącza ograniczenie
  prędkości zależne od lokalnej gęstości. `gawkers` działają w scenariuszu
  etapowym i przy zdarzeniach obserwowalnych.
- `stewards.*` włącza służby przy bramach; `gates: all` obejmuje wszystkie
  bramy, można podać listę identyfikatorów. Przekierowanie ma zastosowanie przy
  `exits.choice.mode: softmax`.
- `hazard.model`: `legacy` oznacza dyfuzję, `advection` uwzględnia wiatr,
  wzrost/wygaśnięcie źródła i opcjonalne blokowanie przez budynki.
  `wind.to_deg` wskazuje kierunek, **dokąd** wieje (0° na wschód, 90° na północ);
  `random_dir: true` losuje kierunek z ziarna.
- `events` to lista zdarzeń (opisana niżej). Pusta lista oznacza zdarzenie
  zgodne ze starszym ustawieniem `hazard`, jeśli `hazard.enabled` jest prawdziwe.
- `exits.closed` zamyka bramy od startu; `close_at_s` podaje mapę
  `ID_bramy: sekundy_od_alarmu` dla zamknięcia w trakcie. `choice.mode`:
  `legacy` wybiera według starszej reguły, `softmax` włącza probabilistyczny
  wybór, wiedzę o bramach, naśladowanie i lokalne wykrywanie zatorów.
- `injury.mode`: `legacy` oznacza starszą regułę przewróceń, `dose` dawkę
  urazu. Progi i dawki nie stanowią walidacji medycznej.
- `barriers` dodaje przeszkody liniowe konfiguracyjnie, bez zmiany JSON mapy.
- `output.video`, `fps`, `render_progress_every`, `trajectory`, `heatmap`,
  `underlay` sterują zapisem i wizualizacją. `render_progress_every` określa,
  co ile wyrenderowanych klatek wypisywać postęp i czas; można go zmienić np.
  przez `--set output.render_progress_every=250`. `underlay` może zawierać
  `{path, extent, alpha}`, gdzie `extent` to `[xmin, xmax, ymin, ymax]`.

## 5. Parametry zdarzeń

Każdy element `events:` ma typ `type` i opcjonalny czas `t_s` liczony od
alarmu. Zdarzenie położeniowe przyjmuje `xy: [x,y]` albo `at`:

- `at: grill` — losowy grill w aktywnej części mapy;
- `at: random` — losowy punkt aktywnej części;
- `at: [[x1,y1], [x2,y2], ...]` — losowanie z listy punktów;
- opcjonalne `region: [xmin,xmax,ymin,ymax]` ogranicza losowanie;
- `label` ustala tekstową nazwę zdarzenia.

| `type` | Parametry zdarzenia |
|---|---|
| `fire` | `t_s`, lokalizacja, `label`. Dym konfiguruje się w sekcji `hazard`. |
| `bang` | `t_s`, lokalizacja, `label`, `panic_peak` (domyślnie 0.7), `scale_m` (40.0), `hear_radius_m` (brak = bez limitu słyszalności). |
| `package` | `t_s`, lokalizacja, `label`, `r0_m` (8.0), `growth_mps` (0.25), `r_max_m` (40.0), `close_gates` (true), `zone_cost` (60.0), `push_mps2` (2.5). |
| `gate_block` | `t_s`, `gate` (ID lub lista ID); blokuje wskazaną bramę w trakcie symulacji. |
| `pa_failure` | Bez dodatkowych wymaganych pól; wyłącza komunikat globalny. |
| `storm` | `t_s`, `label`, `hurry` (1.25), `shelter_bonus_m` (30.0), `panic` (0.15). |

Nieznany typ zdarzenia jest pomijany z ostrzeżeniem. Zestawy nazwane zapisuje
się w `event_presets`, np.:

```yaml
event_presets:
  huk: [{type: bang, t_s: 0, at: random, panic_peak: 0.7, scale_m: 40}]
  blokada: [{type: fire, t_s: 0, at: grill}, {type: gate_block, t_s: 45, gate: [W1, Z1]}]
```

## 6. Pola mapy JSON

Mapa musi zawierać `granica` jako wielokąt w metrach oraz co najmniej jedno
wyjście. Pozostałe obsługiwane pola:

| Pole | Zawartość |
|---|---|
| `nazwa` | Nazwa mapy. |
| `wyjscia`, `przejscia_zamykane` | Listy bram. Każda: `id`, `odcinek: [[x1,y1],[x2,y2]]`; opcjonalnie `szerokosc_m`, `otwarte`, `opis`. Druga lista oznacza przejścia zamykane. |
| `przeszkody` | Elementy z `aktywna` (domyślnie true) i jednym z: `obrys` (wielokąt), `linia` (+ `grubosc_m`), `xy` (+ `promien_m`); opcjonalne `id`, `rodzaj`. |
| `lawki`, `drzewa`, `grille` | Listy obiektów punktowych z `xy`, opcjonalnie `promien_m` i `aktywna`. Ławki, drzewa i grille można globalnie wyłączyć przez `map_options`. |
| `tunele` | Korytarze powiązane z bramą: `id`, `wyjscie`, oraz `obrys` albo `wektor`; opcjonalnie `aktywny`. `covered_exits` wybiera zadaszone przejścia. |
| `bariery` | Odcinki z `from`/`to` (lub `od`/`do`) i opcjonalnym `thickness_m` (lub `grubosc_m`). |
| `chodniki` | Dane pomocnicze/wizualne; nie wpływają na ruch agentów w obecnym modelu. |

`map_extras` uzupełnia brakujące sekcje mapy (tunele, ławki, grille, drzewa,
bariery); dane obecne już w mapie mają pierwszeństwo. Pola mapy `powierzchnie`
i `kontury` mogą występować w pliku, ale nie są używane jako przeszkody przez
loader symulacji.

Opcjonalne pole `flagi` może sterować stanem elementów mapy:
`flagi.wyjscia` nadpisuje `otwarte` dla bram, `flagi.ploty` włącza lub wyłącza
przeszkody po ich `id`, a `flagi.przeszkody` po ich `rodzaj` (np. `drzewo`).
Flagi mają pierwszeństwo przed polami `otwarte` i `aktywna` w elementach.
Generator dodatków `scripts/make_extras.py` wczytuje mapę wskazaną w
`scenarios/etap2.yaml` i uwzględnia te flagi przy obliczaniu dostępnego terenu.

## 7. Ustawienia wymagające zmiany kodu

Lista w sekcji 4 obejmuje wszystkie parametry konfiguracyjne odczytywane z
`DEFAULTS`. Nie każdy literał liczbowy w kodzie jest ustawieniem modelu:
część to progi algorytmiczne, tolerancje numeryczne lub wartości wewnętrzne,
których nie można zmienić przez YAML/CLI. Ważniejsze stałe zakodowane na sztywno
to:

| Stała / reguła | Miejsce |
|---|---|
| Początkowy poziom paniki `PANICKER_P = 0.8` i próg źródła zarażania `PANIC_SOURCE = 0.3`. | `evac/sim.py` |
| Rozstawianie agentów: maks. 80 iteracji rozpychania, promień par 0.8 m, zapas 0.03 m; rozrzut położenia w grupie 0.3–0.9 m razy pierwiastek z liczebności grupy. | `evac/agents.py` |
| Losowy spacer przed alarmem: w każdym kroku wybierane 5% osób, cel do ±6 m, prędkość spaceru losowana z odchyleniem 0.15 m/s i ograniczana do 0.2–1.0 m/s. | `evac/sim.py`, `evac/agents.py` |
| Zapas przestrzenny przy celach gapiów, progi odchodzenia i część ich zachowania. | `evac/alarm.py` |
| Wewnętrzne limity/tolerancje solverów, siatek, rasteryzacji, interpolacji i renderowania. | `evac/engine.py`, `evac/fields.py`, `evac/geometry.py`, `evac/render.py` |

Zmiana tych wartości wymaga edycji odpowiedniego pliku `.py`; nie są to
publiczne opcje uruchomieniowe. W szczególności `mapa_ms.py` ma osobne słowniki
`OTWARTE` i `PRZESZKODY`, ale loader symulacji bierze stan bram/przeszkód z
map JSON oraz konfiguracji, a nie z tych słowników.

## 8. Skąd wziąć bieżącą, dokładną listę

Opcje CLI mogą zostać zmienione wraz z kodem, dlatego przed użyciem sprawdź:

```powershell
python run.py --help
python run_matrix.py --help
python scripts\sensitivity.py --help
python scripts\bottleneck.py --help
```

Wartości domyślne i kolejność nadpisywania znajdują się w
`evac/config.py`; ustawienia konkretnego przebiegu mogą dodatkowo pochodzić
z wybranego pliku scenariusza.
