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

Docelowo projekt end-to-end uruchamiany przez `docker compose up`: GCP (GCS + BigQuery), dbt, Airflow, MLflow, FastAPI, Docker, GitHub Actions, ewentualnie RAG. Na razie etap: notebooki + moduły w `src/`.

## Dane

- **PSE API** `https://api.raporty.pse.pl/api/kse-load` (OData, bez klucza). Filtr `$filter=business_date eq 'YYYY-MM-DD'`, stronicowanie przez `nextLink`. Rekordy co 15 min (96/dobę; 92 i 100 w dni zmiany czasu). Dane dopiero od **2024-06-14**. Kolumny: `dtime` i `dtime_utc` (**koniec** kwadransa), `load_actual` (target, MW), `load_fcst` (prognoza PSE = benchmark), `business_date`.
- **Open-Meteo archive** `https://archive-api.open-meteo.com/v1/archive`: pogoda godzinowa (UTC) dla 8 miast (Warszawa, Kraków, Wrocław, Łódź, Poznań, Gdańsk, Szczecin, Katowice), średnia ważona liczbą ludności. Zmienne: temperature_2m, apparent_temperature, shortwave_radiation, cloud_cover, wind_speed_10m, precipitation.
- Folder `data/` **nie jest w gicie** (.gitignore). Na nowym komputerze odtwarza się go notebookami 02 → 04 → 05.

## Struktura repo

- `src/pse_client.py`: `fetch_day`, `fetch_load(start, end)`.
- `src/weather_client.py`: `fetch_weather`, `weighted_average`, listy `CITIES` i `VARIABLES`.
- `src/features.py`: `build_features(load, weather)`, czyli 1 wiersz = 1 godzina UTC:
  - kwadranse → godziny (cofnięcie o 15 min, bo dtime to koniec kwadransa),
  - kalendarz w czasie Europe/Warsaw (hour, dayofweek, month, is_weekend, is_holiday z biblioteki `holidays`),
  - heating_degrees (15 °C) i cooling_degrees (22 °C),
  - lagi **od 48 h wzwyż** (load_lag_48h, load_lag_168h, load_mean_7d), żeby nie było data leakage.
- `notebooks/`:
  - 01: eksploracja,
  - 02: pobranie historii PSE → `data/raw/kse_load.parquet`,
  - 03: pierwszy rzut oka na pogodę,
  - 04: pobranie pogody → `data/raw/weather_cities.parquet` i `weather_pl.parquet`,
  - 05: tabela cech → `data/processed/features.parquet` + benchmarki naiwne,
  - 06: model LightGBM.

## Aktualne wyniki (stan na 2026-11-23)

Notebook 06, podział w czasie: trening do 2025-09-30, test 2025-10-01 → 2026-11-22. Bias = rzeczywiste − prognoza, więc **plus znaczy, że model zaniża**.

| Wariant | MAE [MW] | bias [MW] |
|---|---|---|
| Prognoza PSE | 420 | −129 |
| LightGBM, douczanie co miesiąc + cel względny (load / load_mean_7d) | 538 | +145 |
| LightGBM, douczanie co miesiąc | 550 | +200 |
| LightGBM jednorazowy | 614 | +272 |
| Naiwna „jak tydzień temu” | ~1000 | |

Wniosek: model systematycznie zaniża zimą (styczeń 2026: średnio −4,3 °C wobec +1,9 °C rok wcześniej, bias ok. +800 MW), bo drzewa nie ekstrapolują poza zakres treningu, a w treningu była tylko jedna łagodna zima.

**Uwaga (uczciwość):** w teście używamy pogody faktycznej, a w produkcji będzie tylko prognoza pogody, więc wyniki są lekko optymistyczne.

## Następne kroki (pomysły)

1. LightGBM z `linear_tree=True`, żeby ekstrapolować zależność od temperatury.
2. Douczanie co tydzień zamiast co miesiąc.
3. Nowe cechy: średnia temperatura z 24 h, dni mostowe i okres świąteczny, długość dnia.
4. MLflow do śledzenia eksperymentów; przeniesienie treningu z notebooka do `src/`.
5. Pogoda prognozowana (Open-Meteo forecast / historical forecast API) do realnej prognozy na jutro.

## Zasady techniczne

- **Git:** commity prosto do `main`, bez gałęzi i pull requestów. Na początku pracy `git pull`, na końcu `git push`. Projekt jest rozwijany na dwóch komputerach z Windows, więc git jest jedynym źródłem prawdy.
- **Środowisko:** `.venv` w folderze projektu (Python 3.12). Aktywacja: `.venv\Scripts\Activate.ps1`. Jeśli PowerShell blokuje skrypty: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **requirements.txt:** tylko bezpośrednie zależności z przypiętymi wersjami (**nigdy** `pip freeze`, bo był konflikt python-dateutil i pakiety tylko pod Windows). Plik zapisywać w **UTF-8** (Notatnik domyślnie zapisuje UTF-16).
- Walidacja zawsze **w czasie** (przeszłość → przyszłość), nigdy losowa. Cechy tylko z informacji znanej w chwili prognozy.
- Czas: wszystko w UTC, kalendarz liczony w Europe/Warsaw.
- Główna metryka: MAE w MW, porównanie z `load_fcst` (PSE) i z prognozą naiwną.
