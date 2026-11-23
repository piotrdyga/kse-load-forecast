"""Budowa tabeli cech: 1 wiersz = 1 godzina (UTC).

Zasada: każda cecha musi być znana w chwili robienia prognozy na jutro (brak data leakage).
Uwaga: pogoda to na razie pogoda FAKTYCZNA. W produkcji będzie tylko prognoza pogody.
"""

import holidays
import pandas as pd

from kse_forecast.config import BENCHMARK, LOCAL_TZ, TARGET

CALENDAR_FEATURES = ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]
SPECIAL_DAY_FEATURES = ["is_bridge_day", "is_christmas_period"]
WEATHER_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
    "heating_degrees",
    "cooling_degrees",
]
LAG_FEATURES = ["load_lag_48h", "load_lag_168h", "load_mean_7d"]

# Zestaw cech z pierwszej wersji modelu oraz zestaw rozszerzony o dni nietypowe (wnioski z EDA).
BASE_FEATURES = CALENDAR_FEATURES + WEATHER_FEATURES + LAG_FEATURES
ALL_FEATURES = BASE_FEATURES + SPECIAL_DAY_FEATURES

HEATING_THRESHOLD_C = 15.0
COOLING_THRESHOLD_C = 22.0


def hourly_load(load: pd.DataFrame) -> pd.DataFrame:
    """Kwadranse PSE -> godziny UTC.

    dtime_utc to KONIEC kwadransa, więc cofamy o 15 minut, żeby mieć jego początek.
    Pracujemy na zegarze UTC, który się nie przestawia, więc każda godzina ma dokładnie
    4 kwadranse, także w dni zmiany czasu.
    """
    df = load.copy()
    df["time"] = pd.to_datetime(df["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    return df.set_index("time")[[TARGET, BENCHMARK]].resample("h").mean()


def _polish_holidays(years: range) -> set:
    return set(holidays.Poland(years=years).keys())


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy kalendarzowe liczone w czasie polskim, bo ludzie żyją według lokalnego zegara."""
    local = df.index.tz_convert(LOCAL_TZ)
    dates = pd.DatetimeIndex(local.date)
    # Rok wcześniej i rok później: dni mostowe na przełomie roku patrzą na sąsiednie dni.
    pl_holidays = _polish_holidays(range(dates.year.min() - 1, dates.year.max() + 2))

    df["hour"] = local.hour
    df["dayofweek"] = local.day_of_week  # 0 = poniedziałek, 6 = niedziela
    df["month"] = local.month
    df["is_weekend"] = (local.day_of_week >= 5).astype(int)
    df["is_holiday"] = dates.map(lambda d: d.date() in pl_holidays).astype(int)
    df["is_bridge_day"] = dates.map(lambda d: _is_bridge_day(d, pl_holidays)).astype(int)
    df["is_christmas_period"] = (
        ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 6))
    ).astype(int)
    return df


def _is_bridge_day(day: pd.Timestamp, pl_holidays: set) -> bool:
    """Dzień mostowy: dzień roboczy, który ma dzień wolny zarówno przed, jak i po sobie.

    Przykłady: piątek po Bożym Ciele (czwartek), poniedziałek przed świętem we wtorek.
    """

    def is_day_off(d: pd.Timestamp) -> bool:
        return d.day_of_week >= 5 or d.date() in pl_holidays

    one_day = pd.Timedelta(days=1)
    return (not is_day_off(day)) and is_day_off(day - one_day) and is_day_off(day + one_day)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zależność od temperatury ma kształt U: ogrzewanie przy zimnie, klimatyzacja przy upale."""
    df["heating_degrees"] = (HEATING_THRESHOLD_C - df["temperature_2m"]).clip(lower=0)
    df["cooling_degrees"] = (df["temperature_2m"] - COOLING_THRESHOLD_C).clip(lower=0)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Zapotrzebowanie z przeszłości. Najkrótszy lag to 48 h.

    Prognozę na jutro robimy dziś rano, więc dla każdej godziny jutra pewne są tylko dane
    sprzed ok. 2 dób. Krótszy lag oznaczałby data leakage.
    Wymaga ciągłego indeksu godzinowego (sprawdzane w build_features).
    """
    y = df[TARGET]
    df["load_lag_48h"] = y.shift(48)  # ta sama godzina 2 dni temu
    df["load_lag_168h"] = y.shift(168)  # ta sama godzina tydzień temu
    df["load_mean_7d"] = y.shift(48).rolling(168).mean()  # średnia z 7 dni kończąca się 48 h temu
    return df


def build_features(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Surowe dane PSE + pogoda dla Polski -> tabela cech gotowa dla modelu."""
    hourly = hourly_load(load)
    full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    if len(hourly) != len(full_range) or hourly[TARGET].isna().any():
        raise ValueError("Godzinowe dane PSE mają dziury, a lagi przesuwane o N wierszy byłyby błędne")

    df = add_lags(hourly)  # lagi liczymy PRZED złączeniem z pogodą, póki indeks jest ciągły
    df = df.join(weather, how="inner")
    df = add_calendar(df)
    df = add_weather_features(df)
    return df.dropna()  # pierwsze ~9 dni nie ma pełnych lagów, więc je odcinamy
