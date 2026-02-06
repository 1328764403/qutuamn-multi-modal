"""
Utility to load A-share CSV data (price / macro / text features)
into the generic multimodal format used by this project.

Expected CSV files in `data_dir` (produced by `download_a_share_data.py`):
- price.csv: columns = ["date", "close", ...]  (at least these two)
- macro.csv: columns = ["date", "macro_dummy", ...]
- text_features.csv: columns = ["date", "txt_dummy", ...]

This loader constructs for each sample a sliding window of length `window`
for all three modalities, and a regression label defined as the future
`lead`-day return based on the close price:

    y_t = close_{t+lead} / close_t - 1

It returns a dict with the same structure as other dataset loaders:

{
  "train": {
    "modalities": [np.array, np.array, np.array],
    "labels": np.array
  },
  "val": { ... },
  "test": { ... }
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_and_merge_csvs(data_dir: str) -> pd.DataFrame:
    """Load price, macro and text CSVs and merge on date."""
    data_path = Path(data_dir)

    price_path = data_path / "price.csv"
    macro_path = data_path / "macro.csv"
    text_path = data_path / "text_features.csv"

    if not price_path.exists():
        raise FileNotFoundError(f"price.csv not found at {price_path}")
    if not macro_path.exists():
        raise FileNotFoundError(f"macro.csv not found at {macro_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"text_features.csv not found at {text_path}")

    price = pd.read_csv(price_path, parse_dates=["date"])
    macro = pd.read_csv(macro_path, parse_dates=["date"])
    text = pd.read_csv(text_path, parse_dates=["date"])

    # Basic column checks with graceful fallbacks
    if "close" not in price.columns:
        raise KeyError(f"'close' column not found in {price_path}")
    if "macro_dummy" not in macro.columns:
        # Allow user to replace macro_dummy with real features later
        raise KeyError(f"'macro_dummy' column not found in {macro_path}")
    if "txt_dummy" not in text.columns:
        raise KeyError(f"'txt_dummy' column not found in {text_path}")

    df = price[["date", "close"]].merge(
        macro[["date", "macro_dummy"]], on="date", how="inner"
    )
    df = df.merge(text[["date", "txt_dummy"]], on="date", how="inner")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def _build_windows_and_labels(
    df: pd.DataFrame, window: int, lead: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows for three modalities and future-return labels.

    Returns:
        X_price: (N, window, 1)
        X_macro: (N, window, 1)
        X_text:  (N, window, 1)
        y:       (N, 1)
        dates:   (N,) pd.Timestamp
    """
    closes = df["close"].astype("float32").to_numpy()
    macro_vals = df["macro_dummy"].astype("float32").to_numpy()
    text_vals = df["txt_dummy"].astype("float32").to_numpy()
    dates_all = df["date"].to_numpy()

    X_price: List[np.ndarray] = []
    X_macro: List[np.ndarray] = []
    X_text: List[np.ndarray] = []
    y: List[List[float]] = []
    dates: List[pd.Timestamp] = []

    # We need a full window ending at i, and a label at i+lead
    for i in range(window - 1, len(df) - lead):
        start = i - window + 1
        end = i + 1  # exclusive

        # Future return based on close price
        ret = float(closes[i + lead] / closes[i] - 1.0)

        X_price.append(closes[start:end, None])  # (window, 1)
        X_macro.append(macro_vals[start:end, None])
        X_text.append(text_vals[start:end, None])
        y.append([ret])
        dates.append(dates_all[i])

    if not X_price:
        raise ValueError(
            f"No samples constructed. Check window={window}, lead={lead}, "
            f"and length of dataframe={len(df)}."
        )

    X_price_arr = np.stack(X_price, axis=0)  # (N, window, 1)
    X_macro_arr = np.stack(X_macro, axis=0)
    X_text_arr = np.stack(X_text, axis=0)
    y_arr = np.stack(y, axis=0).astype("float32")  # (N, 1)
    dates_arr = np.array(dates)

    return X_price_arr, X_macro_arr, X_text_arr, y_arr, dates_arr


def _split_by_dates(
    X_price: np.ndarray,
    X_macro: np.ndarray,
    X_text: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    split_dates: Dict[str, Tuple[str, str]],
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Split data into train/val/test by date ranges defined in split_dates.

    split_dates format (strings in 'YYYY-MM-DD'):
    {
      "train": ["2016-01-01", "2021-12-31"],
      "val":   ["2022-01-01", "2022-12-31"],
      "test":  ["2023-01-01", "2024-12-31"]
    }
    """
    def _mask_for_range(start: str, end: str) -> np.ndarray:
        s = np.datetime64(start)
        e = np.datetime64(end)
        return (dates >= s) & (dates <= e)

    splits = {}
    for split_name in ["train", "val", "test"]:
        if split_name not in split_dates:
            continue
        start, end = split_dates[split_name]
        mask = _mask_for_range(start, end)

        Xp = X_price[mask]
        Xm = X_macro[mask]
        Xt = X_text[mask]
        y_split = y[mask]

        splits[split_name] = {
            "modalities": [Xp, Xm, Xt],
            "labels": y_split,
        }

    return splits


def load_a_share_data(
    data_dir: str,
    window: int = 30,
    lead: int = 1,
    split_dates: Dict[str, Tuple[str, str]] | None = None,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    High-level loader used by train.py when data.source == 'a_share'.

    Args:
        data_dir: directory containing price.csv, macro.csv, text_features.csv
        window: sliding window length (number of past days per sample)
        lead:   prediction horizon (in days) for future return
        split_dates: dict specifying train/val/test date ranges

    Returns:
        dict with 'train', 'val', 'test' splits, each containing:
        - 'modalities': list of np.ndarray, each (N_split, window, 1)
        - 'labels':     np.ndarray, (N_split, 1)
    """
    if split_dates is None:
        raise ValueError("split_dates must be provided for A-share loader.")

    df = _load_and_merge_csvs(data_dir)
    X_price, X_macro, X_text, y, dates = _build_windows_and_labels(df, window, lead)

    splits = _split_by_dates(X_price, X_macro, X_text, y, dates, split_dates)

    # Basic diagnostics
    print("\n=== A-share Data Diagnostics ===")
    for name in ["train", "val", "test"]:
        if name not in splits:
            continue
        mods = splits[name]["modalities"]
        labels = splits[name]["labels"]
        print(
            f"{name.capitalize():>5}: "
            f"N={len(labels)}, "
            f"shapes={[m.shape for m in mods]}"
        )
    return splits

