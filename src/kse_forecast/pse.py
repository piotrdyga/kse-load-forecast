"""Klient API PSE (raport kse-load): pobieranie, przyrostowa aktualizacja i kontrola jakości."""

import logging
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kse_forecast.config import DATA_START, LOCAL_TZ

logger = logging.getLogger(__name__)

BASE_URL = "https://api.raporty.pse.pl/api/kse-load"
TIMEOUT = 30

# Przy aktualizacji pobieramy ponownie kilka ostatnich dni, bo PSE może je jeszcze poprawiać.
REFETCH_DAYS = 3

# Poprawna doba ma 96 kwadransów, a w dni zmiany czasu 92 (marzec) lub 100 (październik).
VALID_QUARTERS_PER_DAY = {92, 96, 100}


def make_session() -> requests.Session:
    """Sesja HTTP, która sama ponawia zapytanie przy chwilowych błędach serwera."""
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_day(day: str, session: requests.Session) -> list[dict]:
    """Wszystkie kwadranse jednej doby (format 'YYYY-MM-DD'), z obsługą stronicowania API."""
    response = session.get(BASE_URL, params={"$filter": f"business_date eq '{day}'"}, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    rows = data["value"]

    # API może podzielić wynik na strony i wtedy podaje adres następnej.
    while next_link := data.get("nextLink"):
        response = session.get(next_link, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        rows.extend(data["value"])
    return rows


def fetch_load(start: str, end: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Dane dla zakresu dni (oba końce włącznie), posortowane po czasie UTC."""
    session = session or make_session()
    rows = []
    for day in pd.date_range(start, end):
        rows.extend(fetch_day(day.strftime("%Y-%m-%d"), session))
    if not rows:
        raise ValueError(f"PSE nie zwróciło danych dla zakresu {start} -> {end}")

    df = pd.DataFrame(rows)
    df["dtime_utc"] = pd.to_datetime(df["dtime_utc"], utc=True)
    return df.sort_values("dtime_utc").reset_index(drop=True)


def validate_load(df: pd.DataFrame) -> None:
    """Kontrakt danych: zatrzymuje pipeline, zamiast po cichu uczyć model na złych danych."""
    problems = []
    if (n := df["dtime_utc"].duplicated().sum()) > 0:
        problems.append(f"{n} zduplikowanych znaczników czasu")
    if (n := df[["load_actual", "load_fcst"]].isna().sum().sum()) > 0:
        problems.append(f"{n} pustych wartości load_actual/load_fcst")

    days = pd.to_datetime(df["business_date"])
    missing = pd.date_range(days.min(), days.max()).difference(days.unique())
    if len(missing) > 0:
        problems.append(f"brakujące dni: {[d.date().isoformat() for d in missing[:10]]}")

    per_day = df.groupby("business_date").size()
    bad = per_day[~per_day.isin(VALID_QUARTERS_PER_DAY)]
    if len(bad) > 0:
        problems.append(f"doby z nieprawidłową liczbą kwadransów: {bad.to_dict()}")

    if problems:
        raise ValueError("Dane PSE nie przeszły kontroli jakości: " + "; ".join(problems))


def update_load(path: Path, end: str | None = None) -> pd.DataFrame:
    """Dociąga brakujące dni do pliku parquet (albo pobiera całość, jeśli pliku nie ma).

    end: ostatni dzień do pobrania, domyślnie wczoraj (czas polski), bo dzisiejsza doba jest niepełna.
    """
    end = end or (pd.Timestamp.now(tz=LOCAL_TZ).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if path.exists():
        existing = pd.read_parquet(path)
        last_day = pd.Timestamp(existing["business_date"].max())
        start = (last_day - pd.Timedelta(days=REFETCH_DAYS - 1)).strftime("%Y-%m-%d")
        existing = existing[existing["business_date"] < start]
    else:
        existing, start = None, DATA_START

    logger.info("PSE: pobieram %s -> %s", start, end)
    new = fetch_load(start, end)
    df = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    df = df.sort_values("dtime_utc").reset_index(drop=True)

    validate_load(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(
        "PSE: zapisano %d wierszy (%s -> %s)", len(df), df["business_date"].min(), df["business_date"].max()
    )
    return df
