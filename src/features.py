"""Budowa tabeli cech: 1 wiersz = 1 godzina (UTC)."""
import holidays
import pandas as pd


def hourly_load(load: pd.DataFrame) -> pd.DataFrame:
    """Kwadranse PSE -> godziny. dtime_utc to KONIEC kwadransa, więc cofamy o 15 min."""
    df = load.copy()
    df["time"] = pd.to_datetime(df["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    return df.set_index("time")[["load_actual", "load_fcst"]].resample("h").mean()


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy kalendarzowe liczone w czasie polskim, bo ludzie żyją wg zegara lokalnego."""
    local = df.index.tz_convert("Europe/Warsaw")
    pl_holidays = holidays.Poland(years=range(local.year.min(), local.year.max() + 1))
    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek          # 0 = poniedziałek, 6 = niedziela
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df["is_holiday"] = [int(d in pl_holidays) for d in local.date]
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zależność od temperatury ma kształt U: ogrzewanie przy zimnie, klimatyzacja przy upale."""
    df["heating_degrees"] = (15 - df["temperature_2m"]).clip(lower=0)
    df["cooling_degrees"] = (df["temperature_2m"] - 22).clip(lower=0)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Zapotrzebowanie z przeszłości. Najkrótszy lag to 48 h, żeby nie używać danych,
    których nie znamy w chwili robienia prognozy na jutro (brak data leakage)."""
    y = df["load_actual"]
    df["load_lag_48h"] = y.shift(48)     # ta sama godzina 2 dni temu
    df["load_lag_168h"] = y.shift(168)   # ta sama godzina tydzień temu
    df["load_mean_7d"] = y.shift(48).rolling(168).mean()  # średnia z 7 dni, kończy się 48 h temu
    return df


def build_features(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = hourly_load(load).join(weather, how="inner")
    df = add_calendar(df)
    df = add_weather_features(df)
    df = add_lags(df)
    return df.dropna()   # pierwsze ~9 dni nie ma pełnych lagów, więc je odcinamy