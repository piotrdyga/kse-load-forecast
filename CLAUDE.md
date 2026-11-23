# CLAUDE.md – kontekst projektu dla Claude

## Sposób pracy

- Komunikacja **po polsku**; kod, nazwy plików, kolumn i commity po angielsku.
- Dostarczaj kompletny, działający kod i zawsze krótko wyjaśniaj, **co robi i dlaczego** (kluczowe decyzje).
- Przy ważnych decyzjach dodaj jednozdaniowe uzasadnienie biznesowe / techniczne.
- Komendy terminala (PowerShell) podawaj pojedynczo, z krótkim wyjaśnieniem.
- Notebooki: dodawaj komórki Markdown z opisem po polsku.
- Rzetelnie raportuj wyniki: nie zakładaj, że coś pobije benchmark, dopóki nie zostanie policzone.
- Przy większych zmianach (struktura repo, usuwanie plików, zmiana podejścia) najpierw zapytaj.
- Jeśli istnieje `CLAUDE.local.md` (lokalny, poza gitem), uwzględnij go.

## Cel projektu

Godzinowa prognoza zapotrzebowania na moc w Krajowym Systemie Elektroenergetycznym (KSE) **na jutro** (24 wartości), porównywana z oficjalną prognozą PSE (`load_fcst`).

Docelowo projekt end-to-end uruchamiany przez `docker compose up`: GCP (GCS + BigQuery), dbt, Airflow, MLflow, FastAPI, Docker, GitHub Actions, ewentualnie RAG. Na razie etap: pakiet `kse_forecast` (pipeline danych, cechy, CV) + notebooki analityczne.

## Dane

- **PSE API** `https://api.raporty.pse.pl/api/kse-load` (OData, bez klucza). Filtr `$filter=business_date eq 'YYYY-MM-DD'`, stronicowanie przez `nextLink`. Rekordy co 15 min (96/dobę; 92 i 100 w dni zmiany czasu). Dane dopiero od **2024-06-14**. Kolumny: `dtime` i `dtime_utc` (**koniec** kwadransa), `load_actual` (target, MW), `load_fcst` (prognoza PSE = benchmark), `business_date`.
- **Open-Meteo archive** `https://archive-api.open-meteo.com/v1/archive`: pogoda godzinowa (UTC) dla 8 miast (Warszawa, Kraków, Wrocław, Łódź, Poznań, Gdańsk, Szczecin, Katowice), średnia ważona liczbą ludności. Zmienne: temperature_2m, apparent_temperature, shortwave_radiation, cloud_cover, wind_speed_10m, precipitation.
- Folder `data/` **nie jest w gicie** (.gitignore). Odtwarza się go poleceniami `python -m kse_forecast update-data` i `python -m kse_forecast build-features`. Pierwsze pobranie trwa kilka minut (ok. 900 zapytań do PSE), kolejne dociągają tylko brakujące dni.

## Struktura repo

- `src/kse_forecast/` (pakiet instalowany przez `pip install -e .`, więc bez `sys.path.append`):
  - `config.py`: ścieżki, strefa czasowa, `HOLDOUT_START` (2026-09-01), `CV_FIRST_TEST_MONTH` (2025-07-01),
  - `pse.py`: klient PSE z ponawianiem zapytań, przyrostowe `update_load`, kontrola jakości `validate_load`,
  - `weather.py`: klient Open-Meteo, `weighted_average` (tylko godziny ze wszystkimi 8 miastami), `update_weather`,
  - `features.py`: `build_features(load, weather)`, czyli 1 wiersz = 1 godzina UTC. Lagi od 48 h wzwyż, liczone przed złączeniem z pogodą. Kalendarz w Europe/Warsaw, w tym `is_bridge_day` i `is_christmas_period` (24.12–6.01). Zestawy cech: `BASE_FEATURES` (16) i `ALL_FEATURES` (18),
  - `validation.py`: `split_holdout`, `monthly_folds` (expanding window, 14 foldów), `fold_data`,
  - `models.py`: kandydaci (`CANDIDATES`) z interfejsem sklearn: `naive_week`, `ridge`, `lgbm_base`, `lgbm`, `lgbm_ratio`, `lgbm_linear_tree`, `ensemble`,
  - `evaluation.py`: `metrics`, `cross_validate` (PSE dołączone jako model `pse`), `summarize`, `fold_mae`,
  - `__main__.py`: CLI `update-data`, `build-features`, `cv`.
- `tests/`: pytest (zmiana czasu, brak leakage w lagach i foldach, dni mostowe, kontrakty danych, średnia pogody).
- `notebooks/`:
  - `01_eda.ipynb`: EDA, **tylko dane deweloperskie** (przed holdoutem), 11 sekcji zakończonych wnioskiem „co z tego wynika dla modelu”,
  - `02_model_selection.ipynb`: walidacja krzyżowa kandydatów, analiza błędów, wybór modelu.
- Stare notebooki (pobieranie danych, pierwszy model z jednym podziałem train/test) usunięte 2026-11-23, są w historii gita.

## Metodologia walidacji

- **Holdout** od 2026-09-01: nie używać w EDA ani przy wyborze modelu i cech. Ocenić na nim model **raz**, na końcu prac.
- **CV:** expanding window, foldy miesięczne 2025-07 … 2026-08 (14). Model trenowany od nowa przed każdym miesiącem na wszystkich wcześniejszych danych.
- Bias = rzeczywiste − prognoza, więc **plus znaczy, że model zaniża**.

## Aktualne wyniki CV (stan na 2026-11-23)

| Model | MAE [MW] | bias [MW] | foldy lepsze od PSE |
|---|---|---|---|
| PSE | 433 | −157 | – |
| **ensemble** (średnia lgbm_ratio + lgbm_linear_tree), wybrany | **496** | +108 | 6/14 |
| lgbm_linear_tree | 531 | +142 | 3/14 |
| lgbm_ratio | 533 | +73 | 4/14 |
| lgbm | 535 | +130 | 2/14 |
| lgbm_base (bez dni mostowych i świątecznych) | 548 | +109 | 2/14 |
| ridge | 666 | +163 | 1/14 |
| naive_week | 990 | +5 | 0/14 |

Wnioski:
- Ok. 60% przewagi PSE pochodzi z zimy (XII–II: ensemble 614 wobec PSE 431). Poza zimą różnica wynosi ok. 30 MW. Styczeń 2026: ensemble ok. 760 MW, pojedyncze LightGBM ok. 1000 MW.
- Przy mrozie < −5 °C modele z elementem liniowym są lepsze od drzew, ale nadal mocno zaniżają (bias ok. +600 MW).
- `is_bridge_day` obniża błąd w dni mostowe z ok. 1650 do ok. 880 MW (lgbm wobec lgbm_base), ale ensemble jest w te dni gorszy (ok. 1270 MW). To kompromis.
- Największa różnica względem PSE jest w nocy (0–7, ok. 100 MW).
- `linear_tree` wymaga celu l2 (l1 nie działa) i mocnej regularyzacji. Bez niej prognozy „wybuchają”. Objective huber z domyślnym alpha=0.9 (w MW) praktycznie nie uczy.
- **Uwaga (uczciwość):** cechy używają pogody faktycznej, a PSE znało tylko prognozę pogody, więc nasze wyniki są zawyżone.

## Kluczowe wnioski z EDA (notebook 01, dane deweloperskie)

- Dane kompletne: brak braków i duplikatów, doby 92/100 kwadransów to dni zmiany czasu.
- Zależność od temperatury jest liniowa przy zimnie: ok. +240 MW/°C poniżej ok. 15 °C, aż do −10 °C. Próg chłodzenia dane wskazują ok. 19 °C (w kodzie 22 °C, różnica RMSE ok. 10 MW).
- Zima 2025/26 dużo mroźniejsza od 2024/25 (min. −15,6 °C wobec −9,8 °C; 513 wobec 87 godzin poniżej −5 °C).
- Brak wyraźnego trendu po usunięciu wpływu temperatury.
- Dni mostowe i okres 24.12–2.01: od −9 do −22% (święto w dzień roboczy średnio −23%).
- W słoneczne letnie południa zapotrzebowanie jest ok. 400 MW niższe mimo wyższej temperatury (PV za licznikiem).
- Autokorelacja: lag 168 h = 0,88, lag 48 h = 0,63, lag 336 h = 0,83.
- PSE: MAE ok. 410 MW, największy błąd w południe (ok. 630 MW), latem i w święta; systematycznie zawyża.

## Następne kroki (pomysły, w kolejności priorytetu)

1. Prognozy pogody zamiast faktycznej (Open-Meteo Historical Forecast API). To warunek uczciwego porównania z PSE.
2. Zima: cechy bezwładności cieplnej (średnia temperatura z 24–72 h), lepszy liniowy komponent temperatury.
3. Dni nietypowe: Wielkanoc, osobna korekta lub model dla dni mostowych i świąt (ensemble traci tu zysk z cech).
4. Strojenie hiperparametrów w tej samej CV (Optuna) i MLflow.
5. Lag DST: lagi liczone w godzinach UTC w tygodniu po zmianie czasu porównują godziny przesunięte o 1 h (drobny efekt).
6. Ocena na holdoucie, potem pipeline produkcyjny (Airflow, FastAPI, GCP, Docker).

## Zasady techniczne

- **Git:** commity prosto do `main`, bez gałęzi i pull requestów. Na początku pracy `git pull`, na końcu `git push`. Projekt jest rozwijany na dwóch komputerach z Windows, więc git jest jedynym źródłem prawdy.
- **Środowisko:** `.venv` w folderze projektu (Python 3.12). Aktywacja: `.venv\Scripts\Activate.ps1`. Jeśli PowerShell blokuje skrypty: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **Instalacja:** `pip install -r requirements.txt`, potem `pip install -e .` (pakiet w trybie edytowalnym).
- **Jakość kodu:** przed commitem `pytest` i `ruff check .` (konfiguracja w `pyproject.toml`).
- **Kernel notebooków:** w VS Code wybrać `.venv`. Domyślny kernel `python3` w systemie wskazuje na `ds_env` z miniforge.
- **requirements.txt:** tylko bezpośrednie zależności z przypiętymi wersjami (**nigdy** `pip freeze`, bo był konflikt python-dateutil i pakiety tylko pod Windows). Plik zapisywać w **UTF-8** (Notatnik domyślnie zapisuje UTF-16).
- Walidacja zawsze **w czasie** (przeszłość → przyszłość), nigdy losowa. Cechy tylko z informacji znanej w chwili prognozy.
- Czas: wszystko w UTC, kalendarz liczony w Europe/Warsaw.
- Główna metryka: MAE w MW, porównanie z `load_fcst` (PSE) i z prognozą naiwną.
