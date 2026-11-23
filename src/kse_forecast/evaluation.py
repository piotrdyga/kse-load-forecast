"""Metryki i walidacja krzyżowa modeli-kandydatów."""

import logging

import numpy as np
import pandas as pd

from kse_forecast.config import BENCHMARK, TARGET
from kse_forecast.models import Candidate
from kse_forecast.validation import Fold, fold_data

logger = logging.getLogger(__name__)

PSE_NAME = "pse"  # pod tą nazwą w wynikach występuje oficjalna prognoza PSE


def metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """MAE (główna metryka), RMSE i bias. Bias = rzeczywiste − prognoza, więc plus = model zaniża."""
    error = y_true - y_pred
    return pd.Series(
        {
            "mae": error.abs().mean(),
            "rmse": np.sqrt((error**2).mean()),
            "bias": error.mean(),
        }
    )


def cross_validate(df: pd.DataFrame, candidates: list[Candidate], folds: list[Fold]) -> pd.DataFrame:
    """Prognozy wszystkich kandydatów we wszystkich foldach, w formacie długim.

    Kolumny wyniku: model, fold, y_true, y_pred (indeks = godzina UTC).
    Prognoza PSE jest dołączona jako dodatkowy „model”, żeby porównywać na tych samych godzinach.
    """
    parts = []
    for fold in folds:
        train, test = fold_data(df, fold)
        parts.append(_frame(test, PSE_NAME, fold.name, test[BENCHMARK]))
        for candidate in candidates:
            model = candidate.make_model().fit(train[candidate.features], train[TARGET])
            y_pred = pd.Series(model.predict(test[candidate.features]), index=test.index)
            parts.append(_frame(test, candidate.name, fold.name, y_pred))
        logger.info("CV: fold %s gotowy (train %d h, test %d h)", fold.name, len(train), len(test))
    return pd.concat(parts)


def _frame(test: pd.DataFrame, model: str, fold: str, y_pred: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"model": model, "fold": fold, "y_true": test[TARGET], "y_pred": y_pred})


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metryki na całym okresie CV oraz rozrzut MAE między foldami, posortowane od najlepszego."""
    overall = predictions.groupby("model").apply(lambda g: metrics(g["y_true"], g["y_pred"]))
    per_fold = fold_mae(predictions)
    overall["mae_fold_std"] = per_fold.std()
    overall["folds_better_than_pse"] = (per_fold.lt(per_fold[PSE_NAME], axis=0)).sum()
    return overall.sort_values("mae").round(1)


def fold_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE w każdym foldzie: wiersze = foldy (miesiące), kolumny = modele."""
    error = (predictions["y_true"] - predictions["y_pred"]).abs()
    return error.groupby([predictions["fold"], predictions["model"]]).mean().unstack()
