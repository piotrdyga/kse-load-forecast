import numpy as np
import pandas as pd

from kse_forecast.config import HOLDOUT_START
from kse_forecast.evaluation import PSE_NAME, cross_validate, summarize
from kse_forecast.models import AveragingRegressor, Candidate, ColumnCopy, RatioTargetRegressor
from kse_forecast.validation import fold_data, monthly_folds, split_holdout


def hourly_frame(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    load = rng.normal(20000, 1000, len(idx))
    return pd.DataFrame(
        {"load_actual": load, "load_fcst": load + 100, "load_lag_168h": load + 500, "load_mean_7d": 20000.0},
        index=idx,
    )


def test_folds_are_consecutive_months_ending_at_holdout():
    folds = monthly_folds()
    assert folds[-1].test_end == HOLDOUT_START
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.test_end == nxt.test_start  # bez dziur i bez nakładania


def test_fold_train_is_strictly_before_test():
    df = hourly_frame("2024-07-01", "2026-11-01")
    for fold in monthly_folds():
        train, test = fold_data(df, fold)
        assert train.index.max() < test.index.min()
        assert test.index.max() < HOLDOUT_START


def test_split_holdout_is_disjoint_and_complete():
    df = hourly_frame("2026-08-01", "2026-10-01")
    dev, holdout = split_holdout(df)
    assert dev.index.max() < HOLDOUT_START <= holdout.index.min()
    assert len(dev) + len(holdout) == len(df)


def test_ratio_target_regressor_recovers_scale():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "load_mean_7d": [100.0, 200.0, 300.0]})
    y = X["load_mean_7d"] * 1.1
    model = RatioTargetRegressor(ColumnCopy("x"), "load_mean_7d").fit(X, y)
    np.testing.assert_allclose(model.predict(X), X["x"] * X["load_mean_7d"])


def test_averaging_regressor_is_mean_of_members():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0]})
    model = AveragingRegressor([ColumnCopy("a"), ColumnCopy("b")]).fit(X, X["a"])
    np.testing.assert_allclose(model.predict(X), [2.0, 4.0])


def test_cross_validate_includes_pse_and_candidates():
    df = hourly_frame("2024-07-01", "2026-09-01")
    candidate = Candidate("naive", ["load_lag_168h"], lambda: ColumnCopy("load_lag_168h"), "")
    folds = monthly_folds()[:2]
    predictions = cross_validate(df, [candidate], folds)

    summary = summarize(predictions)
    assert set(summary.index) == {"naive", PSE_NAME}
    assert summary.loc[PSE_NAME, "mae"] == 100.0
    assert summary.loc["naive", "bias"] == -500.0
