import html as html_module
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from src.svg.rendering import render_svg_or_none, render_svg_to_pil
from src.training.lora.eval import summarize_generation_df


def _safe_filename_stem(row_id) -> str:
    """Map a row id to a short, filesystem-safe filename stem."""
    s = str(row_id).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = s.strip("._") or "unknown_id"
    return s[:200]


def save_render_png_pairs(
    pred_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    id_col: str = "id",
    pred_col: str = "pred_svg",
    target_col: str = "target_svg",
    max_rows: int | None = None,
    output_width: int = 256,
    output_height: int = 256,
) -> list[Path]:
    """Write target and prediction SVG renders as PNGs under ``out_dir/render_pngs``.

    Uses ``render_svg_to_pil`` so dimensions are consistent; failed SVGs become a white canvas
    of the requested size (same behavior as ``render_svg_to_pil`` on error).

    Args:
        pred_df: Subset to export (e.g. notebook ``sample_df`` from percentile buckets—not the
            full holdout unless you intentionally pass it).
        out_dir: Model eval directory (e.g. holdout_best_extended/<model_id>).
        id_col: Column used for filenames.
        pred_col: Predicted SVG string column.
        target_col: Gold SVG string column.
        max_rows: If set, only the first ``max_rows`` rows of ``pred_df`` are exported.
        output_width: Render width in pixels.
        output_height: Render height in pixels.

    Returns:
        List of paths written (target and pred paths interleaved per row).
    """
    dest = Path(out_dir) / "render_pngs"
    dest.mkdir(parents=True, exist_ok=True)
    n = len(pred_df) if max_rows is None else min(int(max_rows), len(pred_df))
    written: list[Path] = []
    for i in range(n):
        rec = pred_df.iloc[i]
        stem = _safe_filename_stem(rec[id_col])
        ts = str(rec.get(target_col, "") or "")
        po = str(rec.get(pred_col, "") or "")
        img_t = render_svg_to_pil(ts, output_width=output_width, output_height=output_height)
        img_p = render_svg_to_pil(po, output_width=output_width, output_height=output_height)
        p_target = dest / f"{stem}_target.png"
        p_pred = dest / f"{stem}_pred.png"
        img_t.save(p_target, format="PNG")
        img_p.save(p_pred, format="PNG")
        written.extend([p_target, p_pred])
    return written


def format_example_label(rec) -> str:
    """Format a compact HTML-safe label for one prediction example.

    Args:
        rec (Mapping | pd.Series): Record containing at least ``id`` and optionally
            ``difficulty_bucket_label`` and ``difficulty_percentile``.

    Returns:
        str: Human-readable label describing the example ID and difficulty metadata.
    """
    pid = html_module.escape(str(rec["id"]))
    bucket = rec.get("difficulty_bucket_label")
    percentile = rec.get("difficulty_percentile", np.nan)

    if bucket is not None:
        bucket = html_module.escape(str(bucket))
        if pd.notna(percentile):
            percentile_txt = f"{100.0 * float(percentile):.1f}%"
            return f"id = {pid} | bucket = {bucket} | percentile = {html_module.escape(percentile_txt)}"
        return f"id = {pid} | bucket = {bucket}"
    return f"id = {pid}"


def display_prediction_summary(pred_df: pd.DataFrame, heading: str | None = None) -> dict:
    """Display a compact summary table for a prediction panel.

    Args:
        pred_df (pd.DataFrame): Generation results DataFrame compatible with
            ``summarize_generation_df``.
        heading (str | None, optional): Optional HTML heading displayed above the summary table.
            Defaults to ``None``.

    Returns:
        dict: Summary metrics dictionary returned by ``summarize_generation_df``.
    """
    summary = summarize_generation_df(pred_df)
    summary_df = pd.DataFrame(
        [
            {
                "render_rate": summary["render_rate"],
                "xml_parse_rate": summary["xml_parse_rate"],
            }
        ]
    )
    if heading:
        display(HTML(f"<h3>{html_module.escape(heading)}</h3>"))
    display(summary_df)
    return summary


def display_cross_model_summary(summary_rows, heading: str = "Cross-finalist summary (loaded finalists only)"):
    """Display a cross-run summary table in notebook output.

    Args:
        summary_rows (Sequence[Mapping]): Rows of per-model summary metrics.
        heading (str, optional): HTML heading shown above the table. Defaults to
            ``"Cross-finalist summary (loaded finalists only)"``.

    Returns:
        pd.DataFrame: DataFrame built from ``summary_rows`` and displayed in the notebook.
    """
    summary_df = pd.DataFrame(summary_rows)
    display(HTML(f"<h3>{html_module.escape(heading)}</h3>"))
    display(summary_df)
    return summary_df


def display_text_comparisons(
    pred_df: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    n_rows: int | None = None,
    preview_chars: int = 16000,
    *,
    pred_col: str = "pred_svg",
    target_col: str = "target_svg",
    left_heading: str = "Target SVG (text)",
    right_heading: str = "Predicted SVG (text)",
):
    """Display side-by-side SVG text comparisons in a notebook.

    Args:
        pred_df (pd.DataFrame): DataFrame containing columns ``pred_col`` and ``target_col``.
        title (str | None, optional): Optional HTML heading displayed above the comparison block.
        subtitle (str | None, optional): Optional HTML paragraph displayed below the title.
        n_rows (int | None, optional): Number of rows to show. ``None`` shows the full DataFrame.
        preview_chars (int, optional): Maximum characters shown from each SVG string.
            Defaults to ``16000``.
        pred_col (str, optional): Prediction column name (e.g. ``pred_svg`` or ``raw_pred``).
        target_col (str, optional): Ground-truth SVG column name.
        left_heading (str, optional): Bold label above the target column.
        right_heading (str, optional): Bold label above the prediction column.

    Returns:
        None: HTML blocks are rendered directly in the notebook output.
    """
    if title:
        display(HTML(f"<h3>{html_module.escape(title)}</h3>"))
    if subtitle:
        display(HTML(f"<p>{subtitle}</p>"))

    rows_to_show = len(pred_df) if n_rows is None else min(n_rows, len(pred_df))
    lh = html_module.escape(left_heading)
    rh = html_module.escape(right_heading)
    for i in range(rows_to_show):
        rec = pred_df.iloc[i]
        ts = str(rec.get(target_col, "") or "").strip()
        po = str(rec.get(pred_col, "") or "")
        ts_e = html_module.escape(ts[:preview_chars]) if ts else "(missing target SVG)"
        po_e = html_module.escape(po[:preview_chars])
        display(HTML(f'<h4 style="margin-top:1em;">{format_example_label(rec)}</h4>'))
        display(
            HTML(
                '<div style="display:flex;gap:16px;border:1px solid #ccc;padding:12px;align-items:flex-start;">'
                f'<div style="flex:1;min-width:0;"><b>{lh}</b><pre style="white-space:pre-wrap;word-break:break-word;font-size:11px;">{ts_e}</pre></div>'
                f'<div style="flex:1;min-width:0;"><b>{rh}</b><pre style="white-space:pre-wrap;word-break:break-word;font-size:11px;">{po_e}</pre></div>'
                "</div>"
            )
        )


def display_rendered_comparisons(
    pred_df: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    n_rows: int | None = None,
    *,
    pred_col: str = "pred_svg",
    target_col: str = "target_svg",
    left_title: str = "Target SVG",
    right_title: str = "Predicted SVG",
):
    """Display rendered target-versus-prediction image pairs in a notebook.

    Args:
        pred_df (pd.DataFrame): DataFrame containing columns ``pred_col`` and ``target_col``.
        title (str | None, optional): Optional heading included above the rendered figures.
        subtitle (str | None, optional): Optional paragraph displayed above the figures.
        n_rows (int | None, optional): Number of rows to render. ``None`` renders all rows.
        pred_col (str, optional): Prediction column name for rendering.
        target_col (str, optional): Ground-truth SVG column name.
        left_title (str, optional): Axis title for the target panel.
        right_title (str, optional): Axis title for the prediction panel.

    Returns:
        None: Matplotlib figures are shown directly via ``plt.show()``.
    """
    if title:
        display(HTML(f"<h3>{html_module.escape(title)}</h3>"))
    if subtitle:
        display(HTML(f"<p>{subtitle}</p>"))

    rows_to_show = len(pred_df) if n_rows is None else min(n_rows, len(pred_df))
    for i in range(rows_to_show):
        rec = pred_df.iloc[i]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        img_t = render_svg_or_none(rec.get(target_col))
        img_p = render_svg_or_none(rec.get(pred_col))
        ax0, ax1 = axes[0], axes[1]
        ax0.axis("off")
        ax1.axis("off")

        if img_t is not None:
            ax0.imshow(img_t)
        else:
            ax0.text(0.5, 0.5, "Target render failed", ha="center", va="center")
        ax0.set_title(left_title)

        if img_p is not None:
            ax1.imshow(img_p)
        else:
            ax1.text(0.5, 0.5, "Prediction render failed", ha="center", va="center")
        ax1.set_title(right_title)

        title_parts = [p for p in [title, format_example_label(rec)] if p]
        fig.suptitle(" | ".join(title_parts), fontsize=11)
        plt.tight_layout()
        plt.show()
