"""Podział danych: zbiór deweloperski (do CV) i holdout (do jednej, końcowej oceny).

Walidacja krzyżowa dla szeregów czasowych (expanding window, inaczej „walk-forward”):

    fold 1:  [train ........]  [test: 2025-07]
    fold 2:  [train ..........] [test: 2025-08]
    ...
    fold N:  [train ....................] [test: 2026-08]   | holdout: od 2026-09 (nie dotykamy)

Zwykła k-krotna CV z losowym podziałem byłaby tu błędem: model uczyłby się na przyszłości
i był sprawdzany na przeszłości, więc wyniki byłyby zawyżone.
"""

from dataclasses import dataclass

import pandas as pd

from kse_forecast.config import CV_FIRST_TEST_MONTH, HOLDOUT_START, LOCAL_TZ


@dataclass(frozen=True)
class Fold:
    """Jeden fold: model uczy się na [początek danych, test_start), testujemy na [test_start, test_end)."""

    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def split_holdout(
    df: pd.DataFrame, holdout_start: pd.Timestamp = HOLDOUT_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(dev, holdout): wszystko przed holdout_start oraz wszystko od holdout_start."""
    return df[df.index < holdout_start], df[df.index >= holdout_start]


def monthly_folds(
    first_test_month: pd.Timestamp = CV_FIRST_TEST_MONTH, end: pd.Timestamp = HOLDOUT_START
) -> list[Fold]:
    """Jeden fold na każdy miesiąc kalendarzowy (granice o północy czasu polskiego)."""
    starts = pd.date_range(first_test_month, end, freq="MS", tz=LOCAL_TZ, inclusive="left")
    return [
        Fold(name=s.strftime("%Y-%m"), test_start=s, test_end=min(s + pd.offsets.MonthBegin(1), end))
        for s in starts
    ]


def fold_data(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) dla jednego foldu. Train zawiera WYŁĄCZNIE dane sprzed testu."""
    train = df[df.index < fold.test_start]
    test = df[(df.index >= fold.test_start) & (df.index < fold.test_end)]
    return train, test
