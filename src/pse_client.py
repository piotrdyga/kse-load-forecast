import requests
import pandas as pd

BASE_URL = "https://api.raporty.pse.pl/api/kse-load"


def fetch_day(day: str) -> list[dict]:
    """Pobiera wszystkie wiersze dla jednej doby (format 'YYYY-MM-DD')."""
    params = {"$filter": f"business_date eq '{day}'"}
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()  # przy błędzie (np. 500) od razu wyjątek
    data = response.json()
    rows = data["value"]

    # API może podzielić wynik na strony: wtedy podaje adres następnej
    next_link = data.get("nextLink")
    while next_link:
        response = requests.get(next_link, timeout=30)
        response.raise_for_status()
        data = response.json()
        rows.extend(data["value"])
        next_link = data.get("nextLink")

    return rows


def fetch_load(start_date: str, end_date: str) -> pd.DataFrame:
    """Pobiera dane dla zakresu dni (włącznie z oboma końcami)."""
    all_rows = []
    for day in pd.date_range(start_date, end_date):
        all_rows.extend(fetch_day(day.strftime("%Y-%m-%d")))

    df = pd.DataFrame(all_rows)
    df["dtime_utc"] = pd.to_datetime(df["dtime_utc"], utc=True)
    return df.sort_values("dtime_utc").reset_index(drop=True)