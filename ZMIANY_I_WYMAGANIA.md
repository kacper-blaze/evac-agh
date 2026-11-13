# Etap 2 – zmiany i wymagania (stan na 12.11.2026, termin 19.11.2026)

Projekt: symulacja ewakuacji Miasteczka Studenckiego AGH (Social Force Model).
Dokument opisuje, **co zmieniamy**, **co jest ustalone**, **co jest moim założeniem do potwierdzenia** i **czego potrzebuję od zespołu**.
Wartości liczbowe oznaczone jako *wstępne* są ilustracyjne i niekalibrowane.

---

## 1. Punkt wyjścia

Działa: SFM (Helbing, Farkas, Vicsek 2000), mapa z JSON-a, 3000 osób, dym (dyfuzja), typy agentów (zwykli / ignorujący 20% / panikujący 5%), zarażanie paniką, wybór bramy z kolejką, wideo MP4, tabele CSV, wykres.
Ograniczenia widoczne w wynikach przebiegu bazowego (stara mapa, wszystkie bramy otwarte):

| problem | objaw |
|---|---|
| ewakuacja zbyt szybka | szczyt przepływu na W2 ≈ 5,6 os/(m·s); w literaturze dla drzwi ok. 1,3–1,9 |
| zator w wąskiej bramie | ok. 290 osób spanikowanych poszło do Z1 (1,35 m); spanikowani ignorują kolejki |
| nierealne ciśnienie | maks. ≈ 24 000 N/m (prędkość pożądana paniki 3 m/s) |
| uproszczone obrażenia | przewrócony wstaje po 20 s (moje założenie) |
| wszyscy ruszają jednocześnie | opóźnienie losowe 0–30 s, bez związku z odległością od zdarzenia |

Uwaga techniczna: kompilacja Numby (JIT) nie została tu przetestowana (brak dostępu do PyPI). Testowany jest wariant NumPy; jądro pętlowe sprawdzone w trybie czystego Pythona (zgodność do 1e-14).

---

## 2. Wymagania ustalone z zespołem

### 2.1 Mapa
- Nowa mapa `mapa_ms.json` (będzie jeszcze aktualizowana → loader ma być odporny na zmiany formatu).
- Wyjścia: **W1–W6** (wszystkie prawdziwe), przejścia zamykane **Z1** (1,0 m) i **Z2** (7,45 m). Razem 8 bram.
- **Płot L1** jest nieprzekraczalny w symulowanych momentach (przejście istnieje, ale jest wtedy zamknięte, więc nie jest oznaczone w pliku). Dzieli teren na część północną (aktywną) i południową (odciętą).
- **Cztery warianty mapy (2×2):**

| wariant | zamknięte bramy | czynne bramy |
|---|---|---|
| normalny | – | W1–W6, Z1, Z2 |
| remont | W2, W4 | W1, W3, W5, W6, Z1, Z2 |
| czwartek | Z1, Z2 | W1–W6 |
| remont + czwartek | W2, W4, Z1, Z2 | W1, W3, W5, W6 |

- Dodatkowe elementy terenu: **grille** (miejsca gromadzenia się studentów; w ewakuacji omijane), **ławki**, **drzewa**; opcjonalnie **podkład** z ortofotomapy pod wykresem (tylko wizualnie).

### 2.2 Zdarzenia
- Dym ma rozprzestrzeniać się bardziej realistycznie i **odstraszać agentów** (omijanie, a nie tylko spowolnienie).
- Dodatkowe zdarzenia obok pożaru (zakres na 19.11 – sekcja 4).

### 2.3 Agenci
- **Stopniowy alarm:** najpierw uciekają osoby najbliżej zdarzenia, dopiero później alarm obejmuje całe miasteczko.
- **Gapie:** część osób zostaje wokół miejsca zdarzenia.
- **Wybór wyjścia:** losowość przy pierwszym wyborze (niekoniecznie najbliższe wyjście); przy zatorze agent zmienia bramę.
- **Znajomość bram:** W6 znana tylko części osób.
- **Przepustowość bram:** realistyczna; obecnie ewakuacja jest zbyt szybka.
- **Obrażenia:** część osób nie wyjdzie z powodu urazów z ewakuacji (ciśnienie, stratowanie, złamania), co dodatkowo opóźnia ewakuację.

### 2.4 Losowość i wyniki
- Losowość miejsc (położenie źródła zdarzenia, rozmieszczenie początkowe) i ziaren; wyniki jako rozkłady (średnia ± przedział), nie pojedynczy przebieg.
- Liczba osób jako jeden parametr: bazowo 3000, porównanie 6000 i 10000.
- Dostarczane: wideo MP4 + tabela statystyk (jak dotychczas).

---

## 3. Zmiany w projekcie (wg etapów)

Zasada: każda zmiana to moduł z przełącznikiem w YAML-u; **wartości domyślne odtwarzają dotychczasowe zachowanie**. Test regresyjny: to samo ziarno na starych ustawieniach daje te same wyniki.

### Etap 1 – mapa i przepustowość (13–14.11)
| zmiana | plik |
|---|---|
| wczytywanie przeszkód typu `linia` (płot) obok `obrys` | `geometry.py` |
| 8 bram; warianty 2×2 jako nakładki w YAML (`variants:`) | `config.py`, `scenarios/` |
| osiągalność: agenci tylko tam, skąd da się dojść do otwartej bramy; raport „wyłączone m²”; gęstość liczona na aktywną część | `geometry.py`, `agents.py`, `stats.py` |
| opcjonalne przejście w płocie (współrzędne + otwarte/zamknięte), domyślnie wyłączone | `config.py`, `geometry.py` |
| test szczelności płotu (siatka 0,2 m vs płot 0,15 m) | `tests/` |
| futryny bram jako ściany, korytarze przejść przez budynek (W1–W3 i inne), obniżenie `v_max` paniki | `geometry.py`, `config.py` |
| test wąskiego gardła vs dane (Jülich) | `scripts/bottleneck.py` |

### Etap 2 – zachowanie (15–16.11)
| zmiana | opis |
|---|---|
| etapowy alarm | stany nieświadomy → zauważył → zdecydował → ucieka; fala informacji od zdarzenia, globalny komunikat z opóźnieniem |
| gapie | osobny typ agenta z punktem obserwacji przy zdarzeniu; odchodzą po dotarciu dymu / spadku liczby gapiów / progu paniki |
| wybór wyjścia | losowość pierwszego wyboru (softmax po kosztach); częściowa znajomość bram (W6 ograniczona); naśladowanie sąsiadów; lokalne wykrywanie zatoru zamiast globalnego liczenia kolejki; stopniowa racjonalność zamiast progu p = 0,6 |

### Etap 3 – obrażenia i dym (17.11)
| zmiana | opis |
|---|---|
| model obrażeń | wskaźnik urazu rośnie z dawki (czas × przekroczenie progu ciśnienia; deptanie leżących zwiększa dawkę); stany: bez obrażeń → lekko ranny (idzie wolniej) → ciężko ranny (leży, nie wstaje); zastępuje „wstaje po 20 s” |
| nowe statystyki | wyszli cało / wyszli ranni / nie wyszli (ciężko ranni w środku); czas do ewakuacji pozostałych |
| dym | adwekcja + dyfuzja (wiatr), źródło zmienne w czasie, blokowanie przez budynki, koszt trasy rosnący ze stężeniem (omijanie), siła odpychająca, strefa nie do wejścia |

### Etap 4 – zdarzenia, losowość, wyniki (18.11)
- Zdarzenia (ogólna lista `events:` w YAML: typ, czas, pozycja, parametry).
- Losowanie źródła zdarzenia i rozmieszczenia początkowego; seria przebiegów z wieloma ziarnami.
- Grille / ławki / drzewa jako przeszkody i punkty startowych skupisk.
- Tabela porównawcza (warianty × liczebności), wideo wybranych przebiegów, aktualizacja dokumentu zespołu.

### 19.11 – bufor i oddanie.

---

## 4. Zdarzenia na termin 19.11

| zdarzenie | efekt w modelu | status |
|---|---|---|
| pożar od grilla | źródło dymu przy grillu | **na 19.11** |
| huk / petarda | fala paniki bez dymu | **na 19.11** |
| podejrzany pakunek | strefa wykluczenia rosnąca z czasem | **na 19.11** |
| awaria nagłośnienia | brak globalnego alarmu, informacja tylko pocztą pantoflową | po terminie (miejsce w architekturze) |
| nagła burza | szukanie schronienia | po terminie |
| blokada bramy w trakcie | rozszerzenie dzisiejszego `close_at_s` | po terminie |

---

## 5. Założenia do potwierdzenia (moje, wstępne)

| parametr | wartość wstępna | uwaga |
|---|---|---|
| znajomość W6 | 15% osób | do ustalenia przez zespół |
| znajomość pozostałych bram | 90–100% | do ustalenia |
| prędkość pożądana paniki | ≈ 1,8–2 m/s (dziś 3,0) | do kalibracji na teście wąskiego gardła |
| prędkość rozchodzenia się informacji | kilka m/s | brak danych, do dyskusji |
| opóźnienie globalnego alarmu | 60–120 s | do dyskusji |
| odsetek gapiów | do ustalenia | |
| progi obrażeń (lekko / ciężko) | do uzupełnienia z literatury | znany tylko próg 1600 N/m (Helbing 2000); wyższe progi to moje założenie, dopóki nie przejrzę źródeł |
| liczba ziaren | ok. 10 dla 3000 osób, ok. 3 dla 10 000 | ograniczenie czasu obliczeń |
| płot L1 | stała przeszkoda w każdym wariancie | zgodnie z ustaleniem |

---

## 6. Czego potrzebuję od zespołu

1. Współrzędne (lub orientacyjne miejsca) **grilli, ławek i drzew** – jeśli ich nie ma, rozmieszczę przykładowo i oznaczę jako przykładowe.
2. Czy położenie źródła zdarzenia ma być stałe (dziś [80, 219], na lewo od toi-toi), czy losowane z zadanego obszaru / listy miejsc.
3. Czy 3000 / 6000 / 10 000 osób dotyczy **aktywnej (północnej) części**, czy całego obrysu.
4. Co dokładnie ma być gotowe na 19.11 (wideo, tabela, dokument, prezentacja?) – od tego zależy, jak głęboko dopracować poszczególne etapy.
5. Licencja i źródło podkładu mapowego (propozycja: ortofotomapa z Geoportalu – do sprawdzenia warunków użycia; pobranie po Waszej stronie).

---

## 7. Kryteria akceptacji (propozycja)

- Nowa mapa wczytuje się; 4 warianty działają; nikt nie zostaje w odciętej części.
- Test regresyjny: stare ustawienia + to samo ziarno = te same wyniki co przed zmianami.
- Przepływ przez bramy w typowych warunkach mieści się w przedziale z literatury (rząd 1–2 os/(m·s)); odchylenia wyjaśnione.
- Ewakuacja jest etapowa: czas ruszenia rośnie z odległością od zdarzenia; gapie obecni.
- Wynik zawiera liczbę poszkodowanych i osób, które nie wyszły; tabela z rozkładami (średnia ± przedział).
- Wideo i tabela dla 4 wariantów × 3 liczebności.

## 8. Ryzyka

- **Czas obliczeń:** przebiegi 10 000 osób są najdłuższe; pełna seria może wymagać zmniejszenia liczby ziaren.
- **Numba:** kompilacja nieprzetestowana w moim środowisku; awaryjnie działa NumPy.
- **Kalibracja:** do 19.11 parametry pozostają ilustracyjne; wyniki ilościowe należy podawać z zastrzeżeniem.
- **Mapa w zmianie:** każda zmiana formatu JSON może wymagać poprawki loadera.
- **Obrażenia:** model stopniowany jest uproszczeniem; nie jest walidowany medycznie.
