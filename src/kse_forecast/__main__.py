"""Wiersz poleceń projektu.

python -m kse_forecast update-data     # dociąga nowe dane z PSE i Open-Meteo
python -m kse_forecast build-features  # buduje tabelę cech
python -m kse_forecast cv              # walidacja krzyżowa wszystkich kandydatów
"""

import argparse
import logging

import pandas as pd

from kse_forecast import config
from kse_forecast.evaluation import cross_validate, summarize
from kse_forecast.features import build_features
from kse_forecast.models import CANDIDATES
from kse_forecast.pse import update_load
from kse_forecast.validation import monthly_folds, split_holdout
from kse_forecast.weather import update_weather, weighted_average

logger = logging.getLogger("kse_forecast")


def cmd_update_data(args: argparse.Namespace) -> None:
    update_load(config.LOAD_RAW_PATH, end=args.end)
    update_weather(config.WEATHER_RAW_PATH, end=args.end)


def cmd_build_features(args: argparse.Namespace) -> None:
    load = pd.read_parquet(config.LOAD_RAW_PATH)
    weather = weighted_average(pd.read_parquet(config.WEATHER_RAW_PATH))
    features = build_features(load, weather)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_PATH)
    logger.info("Cechy: %d wierszy, %s -> %s", len(features), features.index.min(), features.index.max())


def cmd_cv(args: argparse.Namespace) -> None:
    dev, _ = split_holdout(pd.read_parquet(config.FEATURES_PATH))  # holdoutu nie dotykamy
    predictions = cross_validate(dev, CANDIDATES, monthly_folds())
    print(summarize(predictions).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kse_forecast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    update = sub.add_parser("update-data", help="dociąga dane z PSE i Open-Meteo")
    update.add_argument("--end", help="ostatni dzień do pobrania (YYYY-MM-DD), domyślnie wczoraj")
    update.set_defaults(func=cmd_update_data)

    sub.add_parser("build-features", help="buduje tabelę cech").set_defaults(func=cmd_build_features)
    sub.add_parser("cv", help="walidacja krzyżowa kandydatów").set_defaults(func=cmd_cv)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
