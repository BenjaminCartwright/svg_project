"""Notebook 12 helpers: register postprocess variants and score them on a raw-output table."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.svg.cleaning import validate_svg_constraints
from src.training.lora.eval import svg_metrics


def register_method(
    methods: dict[str, Callable[[str], str]],
    summaries: dict[str, str],
    name: str,
    fn: Callable[[str], str],
    summary: str,
) -> None:
    methods[name] = fn
    summaries[name] = summary


def score_postprocess_method(
    name: str,
    fn: Callable[[str], str],
    method_summary: str,
    base_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in base_df.iterrows():
        raw_output = "" if pd.isna(row["raw_output"]) else str(row["raw_output"])
        post_svg = fn(raw_output)
        svg_out = "" if post_svg is None else str(post_svg)
        metrics = svg_metrics(svg_out)
        constraints = validate_svg_constraints(svg_out)
        raw_constraints = validate_svg_constraints(raw_output)
        rows.append(
            {
                "method_name": name,
                "method_summary": method_summary,
                "id": row["id"],
                "prompt": row["prompt"],
                "difficulty_bucket_label": row.get("difficulty_bucket_label", "unknown"),
                "difficulty_percentile": row.get("difficulty_percentile", float("nan")),
                "target_svg": row.get("target_svg", ""),
                "raw_output": raw_output,
                "postprocessed_svg": svg_out,
                "changed_output": raw_output != svg_out,
                "raw_valid_submission": bool(raw_constraints["is_valid_submission_svg"]),
                **metrics,
                **{f"constraint_{k}": v for k, v in constraints.items()},
            }
        )
    return pd.DataFrame(rows)


def pick_gallery_rows(method_df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Prefer renderable rows for matplotlib galleries; fall back if needed."""
    renderable_df = method_df[method_df["render_ok"]].copy()
    if len(renderable_df) >= n_rows:
        return renderable_df.head(n_rows)
    if len(renderable_df) == 0:
        return method_df.head(n_rows)
    needed = n_rows - len(renderable_df)
    fallback_df = method_df[~method_df.index.isin(renderable_df.index)].head(needed)
    return pd.concat([renderable_df, fallback_df], ignore_index=False).head(n_rows)
