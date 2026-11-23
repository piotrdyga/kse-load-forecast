import pandas as pd
import pytest

from kse_forecast.pse import fetch_day, validate_load
from kse_forecast.weather import CITIES, VARIABLES, weighted_average


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Udaje API PSE, które dzieli wynik na dwie strony."""

    def __init__(self):
        self.pages = [{"value": [{"a": 1}], "nextLink": "https://next"}, {"value": [{"a": 2}]}]

    def get(self, url, params=None, timeout=None):
        return FakeResponse(self.pages.pop(0))


def test_fetch_day_follows_pagination():
    assert fetch_day("2025-01-01", FakeSession()) == [{"a": 1}, {"a": 2}]


def pse_day(day: str, n: int = 96) -> pd.DataFrame:
    start = pd.Timestamp(day, tz="Europe/Warsaw").tz_convert("UTC")
    ends = pd.date_range(start, periods=n, freq="15min") + pd.Timedelta(minutes=15)
    return pd.DataFrame({"dtime_utc": ends, "business_date": day, "load_actual": 1.0, "load_fcst": 1.0})


def test_validate_load_accepts_good_data():
    validate_load(pd.concat([pse_day("2025-01-01"), pse_day("2025-01-02")]))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (lambda: pd.concat([pse_day("2025-01-01"), pse_day("2025-01-01")]), "zduplikowanych"),
        (lambda: pd.concat([pse_day("2025-01-01"), pse_day("2025-01-03")]), "brakujące dni"),
        (lambda: pd.concat([pse_day("2025-01-01"), pse_day("2025-01-02", n=95)]), "kwadransów"),
    ],
)
def test_validate_load_rejects_bad_data(data, message):
    with pytest.raises(ValueError, match=message):
        validate_load(data())


def weather_rows(time: str, cities: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"time": pd.Timestamp(time, tz="UTC"), "city": c, **dict.fromkeys(VARIABLES, v)}
        for c, v in cities.items()
    ]
    return pd.DataFrame(rows)


def test_weighted_average_uses_population_weights():
    values = {city: 0.0 for city in CITIES} | {"Warszawa": 10.0}
    result = weighted_average(weather_rows("2025-01-01 00:00", values))
    total = sum(pop for _, _, pop in CITIES.values())
    assert result["temperature_2m"].iloc[0] == pytest.approx(10.0 * CITIES["Warszawa"][2] / total)


def test_weighted_average_skips_hours_with_missing_city():
    complete = weather_rows("2025-01-01 00:00", dict.fromkeys(CITIES, 1.0))
    partial = weather_rows("2025-01-01 01:00", {c: 1.0 for c in CITIES if c != "Warszawa"})
    result = weighted_average(pd.concat([complete, partial]))
    assert list(result.index) == [pd.Timestamp("2025-01-01 00:00", tz="UTC")]
