from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.svg.rendering import render_svg_to_pil


def make_side_by_side_render_figure(
    df,
    target_col: str,
    pred_col: str,
    prompt_col: str,
    n: int = 5,
    figure_title: str = "Rendered SVG Comparison",
):
    """Create a side-by-side qualitative comparison figure.

    Args:
        df (pd.DataFrame): DataFrame containing target SVGs, predicted SVGs, and prompt text.
        target_col (str): Column name holding the reference SVG string.
        pred_col (str): Column name holding the predicted SVG string.
        prompt_col (str): Column name holding the natural-language prompt.
        n (int, optional): Maximum number of rows to display. Defaults to ``5``.
        figure_title (str, optional): Overall matplotlib figure title. Defaults to
            ``"Rendered SVG Comparison"``.

    Returns:
        matplotlib.figure.Figure: Figure with one row per example and two columns showing target
            versus prediction renders.
    """
    n = min(n, len(df))
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 3 * n))

    if n == 1:
        axes = np.array([axes])

    fig.suptitle(figure_title, fontsize=14)

    for i in range(n):
        row = df.iloc[i]

        target_img = render_svg_to_pil(str(row[target_col]))
        pred_img = render_svg_to_pil(str(row[pred_col]))

        axes[i, 0].imshow(target_img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Target #{i+1}")

        axes[i, 1].imshow(pred_img)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Prediction #{i+1}")

        prompt_text = str(row[prompt_col])
        short_prompt = prompt_text[:100] + ("..." if len(prompt_text) > 100 else "")
        axes[i, 0].text(
            0.0, -0.12, f"Prompt: {short_prompt}",
            transform=axes[i, 0].transAxes,
            fontsize=9,
            va="top",
        )

    plt.tight_layout()
    return fig


def render_max_new_tokens_gallery(
    target_svgs: Sequence[str],
    pred_svgs_by_budget: Sequence[Sequence[str]],
    budget_labels: Sequence[str],
    *,
    n_rows: int,
    fig_width: float = 2.8,
    row_height: float = 3.0,
    figure_title: str | None = None,
):
    """One matplotlib figure: ``n_rows`` × (1 + K) images — target then one pred per token budget.

    Args:
        target_svgs: Reference SVG strings, length ≥ ``n_rows``.
        pred_svgs_by_budget: Length K; each entry is a sequence of length ≥ ``n_rows`` aligned
            with ``target_svgs``.
        budget_labels: Length K titles for prediction columns (e.g. ``\"mnt=512\"``).
        n_rows: Number of example rows to draw from the start of each sequence.
        fig_width: Width in inches per column.
        row_height: Height in inches per row.
        figure_title: Optional suptitle.

    Returns:
        matplotlib.figure.Figure
    """
    k = len(pred_svgs_by_budget)
    if len(budget_labels) != k:
        raise ValueError("budget_labels length must match pred_svgs_by_budget")
    n_rows = min(int(n_rows), len(target_svgs))
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    for j in range(k):
        if len(pred_svgs_by_budget[j]) < n_rows:
            raise ValueError(f"pred_svgs_by_budget[{j}] shorter than n_rows")

    ncols = 1 + k
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=ncols,
        figsize=(fig_width * ncols, row_height * n_rows),
    )
    if n_rows == 1:
        axes = np.array([axes])
    if figure_title:
        fig.suptitle(figure_title, fontsize=12)

    for i in range(n_rows):
        axes[i, 0].imshow(render_svg_to_pil(str(target_svgs[i])))
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Target" if i == 0 else "")
        for j in range(k):
            axes[i, j + 1].imshow(render_svg_to_pil(str(pred_svgs_by_budget[j][i])))
            axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(str(budget_labels[j]), fontsize=9)

    plt.tight_layout()
    return fig
