import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert SVG generator. "
    "Given a natural language description, produce a single valid SVG string that matches the prompt. "
    "Return SVG only."
)


def format_svg_instruction_example(prompt_text: str, svg_text: str | None = None, include_answer: bool = True) -> str:
    """Format one instruction-tuning example for causal-LM training.

    Args:
        prompt_text (str): Natural-language image description.
        svg_text (str | None, optional): Target SVG answer appended after the assistant marker
            when ``include_answer`` is ``True``. Defaults to ``None``.
        include_answer (bool, optional): If ``True``, include the SVG target in the formatted
            text; otherwise return a prompt-only inference template. Defaults to ``True``.

    Returns:
        str: Chat-style training string containing system, user, and assistant turns.
    """
    prompt_text = "" if prompt_text is None else str(prompt_text).strip()
    svg_text = "" if svg_text is None else str(svg_text).strip()

    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nPrompt: {prompt_text}\n"
        f"<|assistant|>\n"
    )

    if include_answer:
        text += svg_text

    return text


def build_instruction_dataframe(
    df: pd.DataFrame,
    prompt_col: str,
    svg_col: str,
) -> pd.DataFrame:
    """Create instruction-tuning columns from a raw training DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing prompt and SVG target columns.
        prompt_col (str): Name of the prompt column.
        svg_col (str): Name of the SVG target column.

    Returns:
        pd.DataFrame: Copy of ``df`` with normalized ``prompt``, ``svg_target``, and
            ``train_text`` columns used for LoRA/SFT training.
    """
    out = df.copy()
    out["prompt"] = out[prompt_col].fillna("").astype(str)
    out["svg_target"] = out[svg_col].fillna("").astype(str)
    out["train_text"] = out.apply(
        lambda row: format_svg_instruction_example(
            row["prompt"],
            row["svg_target"],
            include_answer=True,
        ),
        axis=1,
    )
    return out.reset_index(drop=True)
