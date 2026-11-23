"""Pogoda godzinowa z Open-Meteo (archiwum) dla 8 dużych miast i jej średnia ważona dla Polski."""

import logging
from pathlib import Path

import pandas as pd
import requests

from kse_forecast.config import DATA_START, LOCAL_TZ
from kse_forecast.pse import make_session

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 120

# Archiwum Open-Meteo uzupełnia ostatnie dni z opóźnieniem, więc przy aktualizacji
# pobieramy ponownie cały ostatni tydzień.
REFETCH_DAYS = 7

# miasto: (szerokość, długość, ludność w tys.). Ludność (przybliżona) służy jako waga.
CITIES = {
    "Warszawa": (52.23, 21.01, 1860),
    "Krakow": (50.06, 19.94, 800),
    "Wroclaw": (51.11, 17.03, 670),
    "Lodz": (51.76, 19.46, 650),
    "Poznan": (52.41, 16.93, 540),
    "Gdansk": (54.35, 18.65, 490),
    "Szczecin": (53.43, 14.55, 390),
    "Katowice": (50.26, 19.02, 280),
}

VARIABLES = [
    "temperature_2m",  # temperatura [°C]
    "apparent_temperature",  # temperatura odczuwalna [°C]
    "shortwave_radiation",  # nasłonecznienie [W/m²], ważne przez fotowoltaikę
    "cloud_cover",  # zachmurzenie [%]
    "wind_speed_10m",  # wiatr [km/h]
    "precipitation",  # opad [mm]
]


def fetch_city(
    name: str, lat: float, lon: float, start: str, end: str, session: requests.Session
) -> pd.DataFrame:
    """Pogoda godzinowa (UTC) dla jednego miasta."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
    }
    response = session.get(ARCHIVE_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["city"] = name
    return df


def fetch_weather(start: str, end: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Pogoda dla wszystkich miast w jednej długiej tabeli (wiersz = miasto + godzina)."""
    session = session or make_session()
    frames = [fetch_city(name, lat, lon, start, end, session) for name, (lat, lon, _) in CITIES.items()]
    # Ostatnie godziny archiwum bywają jeszcze puste, więc ich nie zapisujemy.
    return pd.concat(frames, ignore_index=True).dropna(subset=VARIABLES, how="all")


def weighted_average(df: pd.DataFrame) -> pd.DataFrame:
    """Jedna „pogoda dla Polski”: średnia z miast ważona liczbą ludności.

    Bierzemy tylko godziny, w których są dane ze WSZYSTKICH miast. Gdyby brakowało np. Warszawy,
    średnia z pozostałych miast po cichu zmieniłaby znaczenie (inne wagi), więc lepiej taką
    godzinę pominąć.
    """
    complete = df.dropna(subset=VARIABLES)
    cities_per_hour = complete.groupby("time")["city"].transform("nunique")
    complete = complete[cities_per_hour == len(CITIES)]

    weights = complete["city"].map({name: pop for name, (_, _, pop) in CITIES.items()})
    weighted = complete[VARIABLES].mul(weights, axis=0)
    weighted["weight"] = weights
    sums = weighted.groupby(complete["time"]).sum()
    return sums[VARIABLES].div(sums["weight"], axis=0)


def update_weather(path: Path, end: str | None = None) -> pd.DataFrame:
    """Dociąga brakujące dni pogody do pliku parquet (albo pobiera całość, jeśli pliku nie ma)."""
    end = end or (pd.Timestamp.now(tz=LOCAL_TZ).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if path.exists():
        existing = pd.read_parquet(path)
        start_ts = existing["time"].max().normalize() - pd.Timedelta(days=REFETCH_DAYS)
        start = start_ts.strftime("%Y-%m-%d")
        existing = existing[existing["time"] < start_ts]
    else:
        existing, start = None, DATA_START

    logger.info("Open-Meteo: pobieram %s -> %s", start, end)
    new = fetch_weather(start, end)
    df = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    df = df.drop_duplicates(subset=["time", "city"], keep="last").sort_values(["city", "time"])

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Open-Meteo: zapisano %d wierszy (do %s)", len(df), df["time"].max())
    return df
