import io
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.inference.generation import generate_svg_prediction
from src.svg.cleaning import validate_svg_constraints
from src.svg.rendering import render_svg_or_none


def svg_metrics(svg_text):
    """Compute parse, render, and submission-validity metrics for one SVG.

    Args:
        svg_text (Any): SVG candidate value, coerced to ``str`` before analysis.

    Returns:
        dict[str, bool | int]: Metrics describing whether the SVG opens/closes correctly, parses
            as XML, renders to an image, satisfies submission constraints, and how many path
            elements it contains.
    """
    svg_text = str(svg_text)
    constraint_metrics = validate_svg_constraints(svg_text)
    out = {
        "has_svg_open": "<svg" in svg_text.lower(),
        "has_svg_close": "</svg>" in svg_text.lower(),
        "xml_parse_ok": False,
        "render_ok": False,
        "pred_char_len": len(svg_text),
        "submission_valid": bool(constraint_metrics["is_valid_submission_svg"]),
        "path_count": int(constraint_metrics["path_count"]),
    }
    try:
        ET.fromstring(svg_text)
        out["xml_parse_ok"] = True
    except Exception:
        out["xml_parse_ok"] = False
    if out["xml_parse_ok"]:
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
            img = Image.open(io.BytesIO(png_bytes))
            img.load()
            out["render_ok"] = True
        except Exception:
            out["render_ok"] = False
    return out


# Column names emitted by ``svg_metrics``; drop before re-scoring to avoid duplicate labels on concat.
SVG_METRIC_COLUMN_NAMES = frozenset(
    {
        "has_svg_open",
        "has_svg_close",
        "xml_parse_ok",
        "render_ok",
        "pred_char_len",
        "submission_valid",
        "path_count",
    }
)


def evaluate_generation_panel(model, tokenizer, val_instruction_df, n_examples=18, seed=42):
    """Generate and score a fixed-size qualitative evaluation panel.

    Args:
        model: Loaded generation model.
        tokenizer: Tokenizer paired with ``model``.
        val_instruction_df (pd.DataFrame): Validation DataFrame containing at least ``prompt`` and
            ``completion`` columns, plus optional difficulty metadata.
        n_examples (int, optional): Maximum number of rows to sample for evaluation. Defaults to
            ``18``.
        seed (int, optional): Random seed used for row sampling. Defaults to ``42``.

    Returns:
        pd.DataFrame: Per-example evaluation DataFrame containing prompts, target SVGs,
            predicted SVGs, difficulty metadata, and metrics from ``svg_metrics``.
    """
    if len(val_instruction_df) == 0:
        return pd.DataFrame()
    panel_df = val_instruction_df.sample(
        n=min(n_examples, len(val_instruction_df)),
        random_state=seed,
    ).reset_index(drop=True)
    rows = []
    for i, row in tqdm(
        panel_df.iterrows(),
        total=len(panel_df),
        desc="Eval panel SVG generation",
    ):
        pred_svg = generate_svg_prediction(row["prompt"], tokenizer, model)
        metrics = svg_metrics(pred_svg)
        rows.append(
            {
                "example_id": i,
                "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
                "prompt": row["prompt"],
                "target_svg": row["completion"],
                "pred_svg": pred_svg,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_generation_df(generation_df):
    """Aggregate panel-level generation metrics into a compact summary.

    Args:
        generation_df (pd.DataFrame): DataFrame returned by ``evaluate_generation_panel`` or any
            compatible table with the same metric columns.

    Returns:
        dict[str, float]: Summary rates and averages. Empty inputs return ``NaN`` values.
    """
    if len(generation_df) == 0:
        return {
            "svg_open_rate": np.nan,
            "svg_close_rate": np.nan,
            "xml_parse_rate": np.nan,
            "render_rate": np.nan,
            "avg_pred_char_len": np.nan,
            "submission_valid_rate": np.nan,
        }
    return {
        "svg_open_rate": float(generation_df["has_svg_open"].mean()),
        "svg_close_rate": float(generation_df["has_svg_close"].mean()),
        "xml_parse_rate": float(generation_df["xml_parse_ok"].mean()),
        "render_rate": float(generation_df["render_ok"].mean()),
        "avg_pred_char_len": float(generation_df["pred_char_len"].mean()),
        "submission_valid_rate": float(generation_df["submission_valid"].mean()),
    }


__all__ = [
    "SVG_METRIC_COLUMN_NAMES",
    "svg_metrics",
    "evaluate_generation_panel",
    "summarize_generation_df",
]
