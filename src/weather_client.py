"""Pobieranie godzinowej pogody z Open-Meteo (archiwum) dla dużych miast Polski."""
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# miasto: (szerokość, długość, ludność w tys. – przybliżona, służy jako waga)
CITIES = {
    "Warszawa": (52.23, 21.01, 1860),
    "Krakow":   (50.06, 19.94, 800),
    "Wroclaw":  (51.11, 17.03, 670),
    "Lodz":     (51.76, 19.46, 650),
    "Poznan":   (52.41, 16.93, 540),
    "Gdansk":   (54.35, 18.65, 490),
    "Szczecin": (53.43, 14.55, 390),
    "Katowice": (50.26, 19.02, 280),
}

VARIABLES = [
    "temperature_2m",        # temperatura [°C]
    "apparent_temperature",  # temperatura odczuwalna [°C]
    "shortwave_radiation",   # nasłonecznienie [W/m²] – ważne przez fotowoltaikę
    "cloud_cover",           # zachmurzenie [%]
    "wind_speed_10m",        # wiatr [km/h]
    "precipitation",         # opad [mm]
]


def fetch_city(name, lat, lon, start, end) -> pd.DataFrame:
    """Pogoda godzinowa (UTC) dla jednego miasta."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["city"] = name
    return df


def fetch_weather(start, end) -> pd.DataFrame:
    """Pogoda dla wszystkich miast w jednej długiej tabeli (wiersz = miasto + godzina)."""
    frames = []
    for name, (lat, lon, _) in CITIES.items():
        print("Pobieram", name)
        frames.append(fetch_city(name, lat, lon, start, end))
    return pd.concat(frames, ignore_index=True)


def weighted_average(df: pd.DataFrame) -> pd.DataFrame:
    """Jedna 'pogoda dla Polski': średnia z miast ważona liczbą ludności."""
    df = df.dropna(subset=VARIABLES)                    # godzin bez danych nie uśredniamy
    w = df["city"].map({n: pop for n, (_, _, pop) in CITIES.items()})
    weighted = df[VARIABLES].mul(w, axis=0)             # wartość × waga
    weighted["w"] = w
    sums = weighted.groupby(df["time"]).sum()
    return sums[VARIABLES].div(sums["w"], axis=0)       # suma(wartość × waga) / suma wag