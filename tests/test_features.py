import numpy as np
import pandas as pd
import pytest

from kse_forecast.features import add_calendar, add_lags, build_features, hourly_load


def quarters(start_utc: str, periods: int) -> pd.DataFrame:
    """Sztuczne dane PSE: kolejne kwadranse, dtime_utc = KONIEC kwadransa."""
    ends = pd.date_range(start_utc, periods=periods, freq="15min", tz="UTC") + pd.Timedelta(minutes=15)
    values = np.arange(periods, dtype=float)
    return pd.DataFrame({"dtime_utc": ends, "load_actual": values, "load_fcst": values + 1})


def test_hourly_load_averages_four_quarters_into_hour_start():
    hourly = hourly_load(quarters("2025-01-15 00:00", 8))
    assert list(hourly.index) == list(pd.date_range("2025-01-15 00:00", periods=2, freq="h", tz="UTC"))
    assert hourly["load_actual"].tolist() == [1.5, 5.5]  # mean(0..3), mean(4..7)


@pytest.mark.parametrize(
    ("polish_day", "start_utc", "n_quarters", "expected_hours"),
    [
        ("2025-10-26", "2025-10-25 22:00", 100, 25),  # cofnięcie zegara: doba ma 25 h
        ("2025-03-30", "2025-03-29 23:00", 92, 23),  # przesunięcie zegara: doba ma 23 h
    ],
)
def test_dst_days_have_correct_number_of_hours(polish_day, start_utc, n_quarters, expected_hours):
    hourly = hourly_load(quarters(start_utc, n_quarters))
    assert len(hourly) == expected_hours
    assert hourly["load_actual"].notna().all()
    local_days = hourly.index.tz_convert("Europe/Warsaw").date
    assert all(d.isoformat() == polish_day for d in local_days)


def test_calendar_uses_polish_local_time():
    # 23:00 UTC w niedzielę zimą to już poniedziałek 0:00 w Polsce
    idx = pd.DatetimeIndex(["2025-01-12 23:00"], tz="UTC")
    cal = add_calendar(pd.DataFrame(index=idx))
    assert cal["hour"].iloc[0] == 0
    assert cal["dayofweek"].iloc[0] == 0


@pytest.mark.parametrize(
    ("day", "is_bridge"),
    [
        ("2025-05-02", 1),  # piątek między 1 maja (czw) a weekendem
        ("2026-06-05", 1),  # piątek po Bożym Ciele
        ("2025-11-10", 1),  # poniedziałek przed 11 listopada (wt)
        ("2025-05-09", 0),  # zwykły piątek
        ("2025-05-03", 0),  # święto w sobotę: dzień wolny, a nie mostowy
    ],
)
def test_bridge_days(day, is_bridge):
    idx = pd.DatetimeIndex([f"{day} 10:00"], tz="Europe/Warsaw").tz_convert("UTC")
    assert add_calendar(pd.DataFrame(index=idx))["is_bridge_day"].iloc[0] == is_bridge


def test_christmas_period():
    days = ["2025-12-23", "2025-12-24", "2025-12-31", "2026-01-06", "2026-01-07"]
    idx = pd.DatetimeIndex([f"{d} 12:00" for d in days], tz="Europe/Warsaw").tz_convert("UTC")
    assert add_calendar(pd.DataFrame(index=idx))["is_christmas_period"].tolist() == [0, 1, 1, 1, 0]


def test_lags_do_not_use_last_48_hours():
    """Brak data leakage: zmiana danych z ostatnich 47 h nie może zmienić cech."""
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    df = pd.DataFrame({"load_actual": np.random.default_rng(0).normal(20000, 1000, 400)}, index=idx)
    lags = add_lags(df.copy())

    t = idx[-1]
    tampered = df.copy()
    tampered.loc[t - pd.Timedelta(hours=47) : t, "load_actual"] = 0.0
    lags_tampered = add_lags(tampered)

    cols = ["load_lag_48h", "load_lag_168h", "load_mean_7d"]
    pd.testing.assert_series_equal(lags.loc[t, cols], lags_tampered.loc[t, cols])
    assert lags.loc[t, "load_lag_48h"] == df.loc[t - pd.Timedelta(hours=48), "load_actual"]


def test_build_features_rejects_gaps_in_load():
    load = quarters("2025-01-01 00:00", 96 * 10)
    load = load.drop(index=range(400, 404))  # wycinamy jedną godzinę
    weather = pd.DataFrame(
        {"temperature_2m": 0.0}, index=pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    )
    with pytest.raises(ValueError, match="dziury"):
        build_features(load, weather)
