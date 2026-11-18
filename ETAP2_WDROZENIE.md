# Etap 2 – co zostało wdrożone (stan na 12.11.2026)

Uzupełnienie dokumentu `ZMIANY_I_WYMAGANIA.md`: co z niego jest zrobione, jak to włączyć, co jest założeniem.
Liczby poniżej pochodzą z pojedynczych przebiegów kontrolnych – **nie są wynikiem serii** i służą tylko do pokazania, że moduły działają.

## 1. Status wymagań

| wymaganie (sekcja) | status | gdzie / jak |
|---|---|---|
| nowa mapa, loader odporny na zmiany (2.1) | zrobione | `geometry.py`; mapa zespołu bez edycji + `mapa_dodatki.json` |
| 8 bram, 4 warianty 2×2 (2.1) | zrobione | `--variant normalny / remont / czwartek / remont_czwartek` |
| płot L1 nieprzekraczalny, część południowa odcięta (2.1) | zrobione | płot jako linia; test szczelności w `tests/test_new_map.py` |
| osiągalność, raport „wyłączone m²”, gęstość na część aktywną | zrobione | aktywne ≈ 11 670 m², wyłączone ≈ 2 990 m² |
| opcjonalne przejście w płocie | zrobione, domyślnie wyłączone | `map_options.fence_gate` |
| grille, ławki, drzewa (2.1) | zrobione; grille i drzewa **przykładowe** | `mapa_dodatki.json`; skupiska startowe `agents.clusters` |
| podkład z ortofotomapy | zrobione (kod); obraz po stronie zespołu | `output.underlay: {path, extent, alpha}` |
| przepustowość bram (2.3) | zrobione | futryny, tunele, `v_max` 1,9, prędkość zależna od gęstości; `scripts/bottleneck.py` |
| stopniowy alarm (2.3) | zrobione | `alarm.mode: staged` |
| gapie (2.3) | zrobione | `behaviour.gawkers` |
| wybór wyjścia: losowość, zmiana przy zatorze, znajomość W6 (2.3) | zrobione | `exits.choice.mode: softmax` |
| obrażenia (2.3) | zrobione; progi = założenie | `injury.mode: dose` |
| dym realistyczniejszy, odstraszający (2.2) | zrobione | `hazard.model: advection` |
| zdarzenia: pożar od grilla, huk, pakunek (4) | zrobione | `events:` / `--event pozar / huk / pakunek` |
| awaria nagłośnienia, blokada bramy w trakcie (4, „po terminie”) | działa, sprawdzone pobieżnie | typy `pa_failure`, `gate_block` |
| nagła burza (4, „po terminie”) | zrobione, sprawdzone pobieżnie | `--event burza` |
| służby porządkowe przy bramach (dodatek) | zrobione, domyślnie wyłączone | `--set stewards.enabled=true` |
| wracanie po znajomych z grupy (dodatek) | zrobione, domyślnie wyłączone | `--set behaviour.group_return.share=0.15` |
| losowość miejsc i ziaren, rozkłady (2.4) | zrobione | `at: grill / random`; `run_matrix.py` → średnia ± 95% CI |
| liczba osób jako jeden parametr (2.4) | zrobione | `agents.count` / `--n` (dotyczy części aktywnej) |
| wideo + tabela (2.4) | zrobione | jak dotąd + nowe kolumny i warstwy |
| test regresyjny (3) | zrobione, przechodzi (różnica 0) | `tests/test_regression.py` |
| Numba nieprzetestowana (8) | sprawdzona – kompiluje się i działa | 3000 osób ≈ 25 s, 10 000 osób ≈ 4–5 min obliczeń na 1 rdzeniu |

**Pełna seria 4 warianty × 3 liczebności × ziarna nie została policzona** – do uruchomienia przez `run_matrix.py`.

## 2. Jak działają nowe moduły (skrót)

**Mapa.** Płot jest rastrowany z promieniem ≥ 0,75 komórki i przedłużony o 0,3 m na końcach, więc jest szczelny także po przekątnej.
Brama z tunelem lub futryną leży na zewnętrznym końcu korytarza – agent musi go przejść, zanim zostanie policzony jako ewakuowany.
Tunele W1–W3 mają kierunek i długość ze starej mapy, a szerokość z aktualnej bramy (W2 ma teraz 5,4 m).

**Przepustowość.** Prędkość pożądana maleje z lokalną gęstością wg diagramu fundamentalnego Weidmanna
(`behaviour.density_speed`). Wynik `python scripts/bottleneck.py`, przepływ jednostkowy [os/(m·s)], literatura ok. 1,3–1,9:

| ustawienia | b=1,0 | 1,5 | 2,5 | 4,0 | 5,4 |
|---|---|---|---|---|---|
| stare, spokojnie | 1,88 | 3,01 | 3,07 | 3,04 | 2,87 |
| stare, panika (3,0 m/s) | 0,93 | 2,60 | 4,18 | 5,13 | 5,84 |
| etap 2, spokojnie | 0,94 | 1,41 | 1,32 | 1,37 | 1,30 |
| etap 2, panika (1,9 m/s) | 1,34 | 1,94 | 1,80 | 1,88 | 1,81 |

Odchylenia: drzwi 1,0 m na siatce 0,2 m wypadają w trybie spokojnym nisko (0,94); w pełnej symulacji chwilowy szczyt 10-sekundowy
na Z1 (1,0 m) sięga ok. 2,3, na pozostałych bramach 0,3–1,9. Źródła literaturowe w skrypcie podane z pamięci – sprawdzić przed cytowaniem.

**Alarm etapowy.** Agent dowiaduje się o zdarzeniu przez: falę informacji (prędkość `info_speed_mps`, zasięg `wave_range_m`),
uciekających sąsiadów, dym, strefę wykluczenia, huk albo komunikat globalny po `global_delay_s`. Nieświadomi dalej spacerują.
Po zauważeniu: zwykli ruszają po `decide_delay_s` + `prep_delay_s`, panikujący od razu, ignorujący czekają na większość sąsiadów.

**Gapie.** Osobny typ; jeśli zauważą pożar lub pakunek z odległości ≤ `max_dist_m`, idą na pierścień obserwacji.
Odchodzą, gdy dotrze dym, wzrośnie ich panika, gapiów ubędzie poniżej ułamka szczytu albo minie `max_watch_s`.

**Wybór wyjścia.** Pierwszy wybór losowy (softmax po koszcie trasy); bramy nieznane są pomijane, ale można je poznać z bliska
albo od sąsiadów. Kto długo stoi w tłoku daleko od bramy, dolicza jej karę i wybiera ponownie. Kolejkę widzi tylko ten, kto jest blisko bramy.
Waga informacji o kolejkach to (1−p)^γ zamiast progu p = 0,6.

**Obrażenia.** Dawka = czas × przekroczenie progu nacisku, plus dawka za upadek i za deptanie leżącego.
Lekko ranny idzie wolniej; ciężko ranny leży do końca i blokuje przejście. Statystyki: wyszli cało / wyszli ranni / nie wyszli.
Próg i dawka krytyczna są dopasowane do dwóch punktów podawanych w literaturze o zgnieceniach w tłumie
(Fruin 1993; Hopkins i in. 1993, za testami Evansa i Haydena): ok. 6227 N na klatkę piersiową → zgon po ok. 15 s,
ok. 1112 N → po 4–6 min (przyjęto 300 s). Dla D = t × (F − F₀) oba punkty dają F₀ ≈ 843 N i D ≈ 80 800 N·s;
po podzieleniu przez obwód ciała (2π·0,3 m, jak w definicji ciśnienia Helbinga): próg ≈ 450 N/m, dawka krytyczna ≈ 43 000 N/m·s.
**Liczby te podaję z pamięci, bez dostępu do źródeł – trzeba je sprawdzić w oryginale przed cytowaniem.**
Zastrzeżenia: dotyczą statycznego ucisku klatki piersiowej, a w SFM „ciśnienie” to suma sił kontaktowych wokół ciała,
próbkowana co 0,25 s; próg „lekko ranny” (¼ dawki krytycznej) pozostaje założeniem własnym.
Przewrócenie nadal następuje po przekroczeniu 1600 N/m (Helbing 2000) przez 0,5 s.

**Dym.** Adwekcja z wiatrem + dyfuzja + zanik, źródło rosnące w czasie, blokowany przez ściany i toi-toie (płot przepuszcza).
Trasy są co `nav_update_s` przeliczane z kosztem rosnącym ze stężeniem; dodatkowo siła odpychająca i strefa „nie do wejścia”.

**Zdarzenia.** Pożar = źródło dymu; huk = skok paniki malejący z odległością, bez dymu; pakunek = rosnąca strefa wykluczenia,
która wypycha ludzi, podnosi koszt tras i zamyka bramy w swoim zasięgu. Burza = wszyscy od razu szukają schronienia:
śpieszą się (prędkość × 1,25) i wolą zadaszone przejścia przez budynki (W1–W3).

**Służby porządkowe (dodatek).** Przy wskazanych bramach: osoby w zasięgu 25 m poznają wszystkie bramy, panika opada,
a idący do tej bramy mogą zostać (raz) przekierowani tam, gdzie łączny czas dojścia i oczekiwania jest krótszy
(służby znają obciążenie wszystkich bram). Wymaga `exits.choice.mode: softmax`.

**Wracanie po znajomych (dodatek).** Część osób z grup, zamiast uciekać, idzie najpierw pod prąd po kolegę, który
jeszcze stoi albo leży (do 30 m); po dojściu kolega rusza razem z nim; po 40 s rezygnuje.

## 3. Decyzje i założenia do potwierdzenia przez zespół

| temat | przyjęte | uwaga |
|---|---|---|
| zasięg fali informacji | 60 m | bez ograniczenia fala 3 m/s obejmowała wszystkich przed komunikatem globalnym |
| huk | słyszą wszyscy od razu | czas ruszenia nie zależy wtedy od odległości; ograniczenie: `hear_radius_m` |
| atrakcyjność szerokiej bramy | `width_bonus_m: 10` | dodatek własny; bez niego tłum blokował się na stałe przy Z1 (1 m) |
| dolna granica racjonalności | 0,15 | dodatek własny; nawet spanikowany w końcu rezygnuje z zatkanej bramy |
| futryny W4–W6, Z1, Z2 | głębokość 0,8 m | do zmiany, jeśli znana jest grubość ścian |
| progi dawki urazu | próg 450 N/m, ciężko 43 000 N/m·s, lekko 10 000 | dopasowane do literatury cytowanej z pamięci (sekcja 2) – **do weryfikacji w źródłach**; „lekko” = założenie |
| miejsce zdarzenia | losowe: grill (pożar) albo dowolny punkt (huk, pakunek) | potwierdzone: ma być losowe |
| zanik dymu | 0,02 1/s | założenie (dym unosi się do góry) |
| odsetek gapiów | 5% | do ustalenia |
| znajomość bram | W6 15%, pozostałe 95% | wartości wstępne z dokumentu |
| grille (6) i drzewa (30) | rozmieszczenie przykładowe | podmienić w `mapa_dodatki.json` lub dodać do mapy zespołu |
| zamknięta brama | wszyscy wiedzą, że jest zamknięta | uproszczenie: nikt nie idzie sprawdzić |

## 4. Przebiegi kontrolne (pożar od grilla, ziarno 42, aktualne ustawienia)

| | 3000, normalny | 6000, remont + czwartek | 10 000, remont + czwartek |
|---|---|---|---|
| ewakuowani | 2999 | 5996 | 9834 (limit 900 s) |
| T90 [s] | 171 | 399 | 695 |
| wyszli ranni | 4 | 106 | 837 |
| nie wyszli (ciężko ranni) | 1 | 4 | 149 |
| przewróceni choć raz | 6 | 142 | 1122 |
| szczyt przepływu [os/(m·s)] | 2,0 | 2,65 | 2,65 |
| czas obliczeń [s] (1 rdzeń, bez wideo) | 25 | ok. 90 | 260 |

Korelacja rang czasu ruszenia z odległością od zdarzenia: 0,95–0,97 (ewakuacja etapowa). Przykładowe pliki: `results_przyklad_etap2/`.
Przy 10 000 osób i czterech czynnych bramach gęstość początkowa to 0,86 os/m²; ok. 10% osób zostaje poszkodowanych –
liczba ta zależy wprost od progów dawki (sekcja 2) i od jednego ziarna, więc traktować ją jako ilustrację.

## 5. Dodatki poza dokumentem wymagań
* mapa zatorów (`*_mapa_zatorow.png`): maks. gęstość i ciśnienie w komórkach 2 × 2 m,
* analiza wrażliwości (`scripts/sensitivity.py`),
* tabela `*_ruszenie_vs_odleglosc.csv` do kryterium „czas ruszenia rośnie z odległością”.

* scenariusze dodatkowe: burza, służby porządkowe, wracanie po znajomych (opis w sekcji 2).

### Scenariusze dodatkowe – pojedyncze przebiegi (pożar, ziarno 42, nowe progi dawki)
| | T90 [s] | poszkodowani | maks. ciśnienie [N/m] | uwagi |
|---|---|---|---|---|
| 3000, normalny, baza | 171 | 5 | 19 166 | |
| 3000, burza | 113 | 11 | 9 351 | 71% osób wychodzi przez W1–W3 |
| 3000, wracanie po znajomych 15% | 166 | 7 | 17 278 | 96 osób wracało |
| 3000, służby przy wszystkich bramach | 201 | 0 | 1 594 | 1071 przekierowanych |
| 6000, remont + czwartek, baza | 399 | 110 | 14 557 | W6: 96 osób |
| 6000, remont + czwartek, służby | 544 | 61 | 4 146 | W6: 624 osoby |

W tych przebiegach służby wyraźnie zmniejszają tłok i liczbę poszkodowanych, ale wydłużają ewakuację, bo kierują ludzi
do dalszych bram. To po jednym ziarnie na wiersz – przy rozrzucie między ziarnami rzędu kilkudziesięciu sekund
wniosek wymaga serii (`run_matrix.py ... --set stewards.enabled=true`).

---

## 6. Poprawka realizmu ciśnienia i przepływu (13.11.2026)

**Diagnoza.** Wartość 19 kN/m z przebiegu kontrolnego nie pochodziła z tłumu. Na 1,7 mln próbek ciśnienia (3000 osób, 150 s) tylko 31 przekraczało 5000 N/m,
a 99,9 percentyl wynosił ok. 300 N/m. Wszystkie skrajne wartości dawały 1–3 osoby tkwiące w ostrych klinach przy ścianach (przy Z1 i narożniku budynku):
siła ścian przeskakiwała tam między dwiema najbliższymi komórkami siatki (do ±150 kN), osoba drgała z prędkością na ograniczniku (5 m/s)
i przez cały przebieg dawała ciśnienie rzędu 17 kN/m (ok. 36 kN na osobę; literatura: 6,2–6,9 kN to zgon lub złamania). To zawyżało liczbę urazów.

**Zmiany (domyślnie włączone, wszystkie z kluczem w konfiguracji):**

| zmiana | klucz | wartość | uzasadnienie |
|---|---|---|---|
| tłumienie normalne kontaktu (osoba–osoba, osoba–ściana) | `sfm.contact_damping_ratio` | 0,5 krytycznego | ciało pochłania energię; likwiduje drgania, które napędzały piki |
| ściśnięcie liczone do ciśnienia ograniczone | `sfm.pressure_overlap_cap_m` | 0,05 m | ok. 20% głębokości klatki (Viano i King 2000), powyżej kości pękają |
| ciśnienie wygładzone w czasie (EMA) dla urazów | `injury.pressure_tau_s` | 1 s | uraz zależy od obciążenia utrzymanego (Hopkins 1993), nie od kroku całkowania |
| górne ograniczenie ciśnienia w dawce | `injury.pressure_cap_Npm` | 3700 N/m (ok. 6,9 kN) | obciążenie złamań kości (Atkinson 1998, za Stillem); wyżej dawka nie rośnie szybciej |
| ogranicznik prędkości | `engine.speed_cap_mps` | 3,0 (było 5,0) | nikt w tłumie nie biegnie 5 m/s |
| osoby w komórce niedostępnej przesuwane do wolnego miejsca | `map_options.unstick` | true | zabezpieczenie; w przebiegach kontrolnych nie było potrzebne (0 przesunięć) |
| szczeliny węższe niż ciało stają się ścianą | `map_options.sliver_clearance_m` | 0,35 m | ślepe kąty, w które nie zmieści się najmniejsza osoba |
| zadaszone przejścia (burza) | `map_options.covered_exits` | W1–W3 | mapa v2 ma korytarze przy W1–W6; bez tego burza nie rozróżniałaby bram |

W statystykach `max_cisnienie_Npm` to teraz wartość wygładzona, a `max_cisnienie_chwilowe_Npm` (ograniczone ściśnięciem 5 cm) zostaje obok.
Test regresyjny uruchamia stary kod z wyłączoną nową fizyką (`tests/_legacy_run.py`), więc nadal pilnuje, że stary tryb się nie zmienił.

**Przebiegi kontrolne po zmianach (mapa v2, ziarno 42, backend NumPy; pojedyncze przebiegi, nie seria):**

| | przed (mapa v1) | po |
|---|---|---|
| 3000, normalny: maks. ciśnienie [N/m] | 19 166 | 3 634 |
| 3000, normalny: T90 [s] / poszkodowani | 171 / 5 | 191 / 9 |
| 6000, remont+czwartek: maks. ciśnienie [N/m] | 14 557 | 5 080 |
| 6000, remont+czwartek: T90 [s] / poszkodowani | 399 / 110 | 346 / 51 |
| przepływ jednostkowy, drzwi 1,0–5,4 m, spokojnie / panika [os/(m·s)] | 0,94–1,41 / 1,34–1,94 | 1,08–1,50 / 1,50–2,07 |

Uwagi: porównanie „przed/po” miesza zmianę mapy (v1 → v2) ze zmianą fizyki, więc nie rozdziela ich wpływu. Przepływ przy panice dla b = 1,5 m (2,07)
jest nieco powyżej zakresu 1,3–1,9 z literatury. Pozostałe punkty podwyższonego ciśnienia (do ok. 3,5 kN/m) leżą przy Z1 (1,2 m), gdzie tłoczy się tłum.
Wariant Numba nie był uruchamiany (brak numby w środowisku); jądro pętlowe sprawdzono w trybie czystego Pythona (`tests/test_backends.py`).
Progi dawki urazu (450 N/m, 43 000 N/m·s) pozostają założeniem własnym dopasowanym do dwóch punktów Hopkinsa i nie zostały zmienione.

### 6a. Progi urazów oparte na literaturze (13.11.2026, wieczór)
* **Lekki uraz** = wygładzone ciśnienie ≥ 1600 N/m (`injury.rib_fracture_Npm`), czyli ok. 3 kN na klatkę: ok. 20% ściśnięcia, ryzyko złamania żeber (Viano i King 2000, za Stillem). Zastępuje własne założenie „¼ dawki krytycznej”.
* **Ciężki uraz** = dawka dopasowana do dwóch szacunków Hopkinsa i in. 1993 (6227 N → 15 s, 1112 N → 4–6 min); bez zmian.
* **Zadaszone przejścia** (burza): wszystkie W1–W6 (`map_options.covered_exits`), zgodnie z ustaleniem zespołu.
* Przebieg kontrolny 6000, remont+czwartek, ziarno 42: T90 332 s, maks. ciśnienie 3810 N/m, 103 lekko rannych, 0 ciężko rannych (po zmianie progu „lekko” z dawki na próg żeber liczba lekko rannych wzrosła z 48 do 103; pojedynczy przebieg).
* Ograniczenia: liczby Hopkinsa i Viano/Kinga znam ze streszczeń (Still), nie z oryginałów; dotyczą statycznego ucisku zdrowych dorosłych. Dopasowanie liniowe t × (F − F₀) przez dwa punkty jest uproszczeniem własnym.
