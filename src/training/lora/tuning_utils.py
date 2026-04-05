"""Helpers for selecting winning hyperparameter runs from tuning result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pick_winner_by_eval_loss(results_df: pd.DataFrame) -> pd.Series:
    """Return the row with lowest ``eval_loss`` (finite); tie-break by row order."""
    if results_df is None or len(results_df) == 0:
        raise ValueError("results_df is empty")
    df = results_df.copy()
    if "eval_loss" not in df.columns:
        raise ValueError("results_df must contain eval_loss column")
    losses = pd.to_numeric(df["eval_loss"], errors="coerce")
    finite = losses[np.isfinite(losses)]
    if len(finite) == 0:
        raise ValueError("No finite eval_loss values")
    idx = finite.idxmin()
    return df.loc[idx]


def append_round_results_csv(path: Path | str, row: dict) -> pd.DataFrame:
    """Append one dict row to CSV (create with header if missing)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if p.is_file():
        old = pd.read_csv(p)
        out = pd.concat([old, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_csv(p, index=False)
    return out
