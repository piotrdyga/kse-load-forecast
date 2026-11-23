"""Modele-kandydaci. Każdy ma interfejs scikit-learn (fit/predict), więc CV traktuje je jednakowo."""

from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kse_forecast.features import ALL_FEATURES, BASE_FEATURES

RANDOM_STATE = 42

LGBM_PARAMS = {
    "objective": "l1",  # minimalizuje błąd bezwzględny, czyli naszą metrykę (MAE)
    "n_estimators": 1000,  # liczba drzew
    "learning_rate": 0.03,  # jak mały krok robi każde kolejne drzewo
    "num_leaves": 63,  # jak szczegółowe może być jedno drzewo
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


class ColumnCopy(BaseEstimator, RegressorMixin):
    """Model naiwny: prognoza = wartość jednej kolumny (np. „tyle, co tydzień temu”)."""

    def __init__(self, column: str = "load_lag_168h"):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy()


class RatioTargetRegressor(BaseEstimator, RegressorMixin):
    """Uczy się proporcji y / X[scale_column] zamiast wartości w MW, a potem mnoży wynik z powrotem.

    Model przewiduje wtedy „o ile procent więcej lub mniej niż ostatnio”, co łagodzi problem
    drzew z wartościami spoza zakresu treningu (np. wyższy ogólny poziom zapotrzebowania).
    """

    def __init__(self, estimator=None, scale_column: str = "load_mean_7d"):
        self.estimator = estimator
        self.scale_column = scale_column

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimator_ = clone(self.estimator).fit(X, y / X[self.scale_column])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X) * X[self.scale_column].to_numpy()


class AveragingRegressor(BaseEstimator, RegressorMixin):
    """Ensemble: średnia prognoz kilku modeli.

    Pomaga, gdy modele mylą się w różnych miejscach (np. drzewa przy mrozie, modele liniowe
    w nietypowe dni), bo błędy częściowo się wtedy znoszą.
    """

    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([est.predict(X) for est in self.estimators_], axis=0)


def make_ridge(features: list[str]):
    """Regresja liniowa (Ridge): godzina, dzień tygodnia i miesiąc jako kategorie (one-hot)."""
    categorical = [c for c in ["hour", "dayofweek", "month"] if c in features]
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return make_pipeline(preprocess, Ridge(alpha=1.0))


@dataclass(frozen=True)
class Candidate:
    """Model-kandydat: nazwa, zestaw cech i funkcja tworząca świeży, nienauczony model."""

    name: str
    features: list[str]
    make_model: Callable[[], BaseEstimator]
    description: str


def make_lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LGBM_PARAMS)


def make_lgbm_ratio() -> RatioTargetRegressor:
    return RatioTargetRegressor(make_lgbm(), "load_mean_7d")


def make_lgbm_linear_tree() -> lgb.LGBMRegressor:
    # linear_tree nie działa z celem l1, więc używamy l2. Bez mocnej regularyzacji (linear_lambda,
    # mniej liści, więcej obserwacji w liściu) prognozy w niektórych miesiącach „wybuchały”
    # (RMSE > 4000 MW). Objective huber z domyślnym alpha=0.9 też się nie nadaje: alpha jest
    # w jednostkach celu (MW), więc gradient byłby obcięty i model prawie by się nie uczył.
    return lgb.LGBMRegressor(
        **{
            **LGBM_PARAMS,
            "objective": "l2",
            "linear_tree": True,
            "linear_lambda": 100.0,
            "num_leaves": 15,
            "min_child_samples": 200,
        }
    )


CANDIDATES = [
    Candidate(
        "naive_week",
        ["load_lag_168h"],
        lambda: ColumnCopy("load_lag_168h"),
        "Naiwny: tyle, co o tej samej godzinie tydzień temu",
    ),
    Candidate(
        "ridge",
        ALL_FEATURES,
        lambda: make_ridge(ALL_FEATURES),
        "Regresja liniowa (Ridge), kalendarz jako one-hot",
    ),
    Candidate(
        "lgbm_base",
        BASE_FEATURES,
        make_lgbm,
        "LightGBM, cechy z pierwszej wersji (bez dni mostowych i świątecznych)",
    ),
    Candidate("lgbm", ALL_FEATURES, make_lgbm, "LightGBM + dni mostowe i okres świąteczny"),
    Candidate("lgbm_ratio", ALL_FEATURES, make_lgbm_ratio, "LightGBM, cel względny: load / load_mean_7d"),
    Candidate(
        "lgbm_linear_tree",
        ALL_FEATURES,
        make_lgbm_linear_tree,
        "LightGBM z liniowymi liśćmi (umie ekstrapolować, np. przy mrozie)",
    ),
    Candidate(
        "ensemble",
        ALL_FEATURES,
        lambda: AveragingRegressor([make_lgbm_ratio(), make_lgbm_linear_tree()]),
        "Średnia z lgbm_ratio i lgbm_linear_tree",
    ),
]
