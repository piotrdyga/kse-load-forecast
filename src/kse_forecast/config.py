"""Stałe projektu w jednym miejscu: ścieżki, strefa czasowa, podział danych.

Dzięki temu żaden moduł ani notebook nie ma wpisanych na sztywno ścieżek typu "../data/raw".
"""

from pathlib import Path

import pandas as pd

# src/kse_forecast/config.py -> parents[2] to katalog główny projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # dane dokładnie tak, jak przyszły z API
PROCESSED_DIR = DATA_DIR / "processed"  # dane przetworzone, gotowe dla modelu

LOAD_RAW_PATH = RAW_DIR / "kse_load.parquet"
WEATHER_RAW_PATH = RAW_DIR / "weather_cities.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

LOCAL_TZ = "Europe/Warsaw"
DATA_START = "2024-06-14"  # od tego dnia API PSE udostępnia dane kse-load

TARGET = "load_actual"  # to prognozujemy [MW]
BENCHMARK = "load_fcst"  # prognoza PSE [MW]: tylko do porównania, NIGDY jako cecha modelu

# --- Podział danych -------------------------------------------------------------------------
# Holdout: ostatni fragment danych odłożony na jedną, końcową ocenę wybranego modelu.
# Nie używamy go ani w EDA, ani przy wyborze modelu i parametrów.
HOLDOUT_START = pd.Timestamp("2026-09-01", tz=LOCAL_TZ)

# Walidacja krzyżowa (expanding window): każdy miesiąc od CV_FIRST_TEST_MONTH do holdoutu jest
# raz zbiorem testowym, a model uczy się na wszystkim, co było przed nim.
# Pierwszy fold ma ok. 12 miesięcy historii, czyli model widzi pełny cykl roczny.
CV_FIRST_TEST_MONTH = pd.Timestamp("2025-07-01", tz=LOCAL_TZ)
