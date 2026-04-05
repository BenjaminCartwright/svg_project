import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def choose_first_existing(dataframe, candidates, name):
    """Return the first candidate column present in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to inspect.
        candidates (Sequence[str]): Ordered column names to try.
        name (str): Human-readable name used in the error message if no column matches.

    Returns:
        str: The first column name from ``candidates`` that exists in ``dataframe``.

    Raises:
        ValueError: If none of the candidate columns are present.
    """
    for col in candidates:
        if col in dataframe.columns:
            return col
    raise ValueError(f"None of {candidates} found in {name}. Columns were: {list(dataframe.columns)}")


def prepare_seq2seq_dataframe(df: pd.DataFrame, prompt_col: str, svg_col: str) -> pd.DataFrame:
    """Clean prompt/SVG columns and drop rows with blank training text.

    Args:
        df (pd.DataFrame): Source DataFrame containing prompt and SVG columns.
        prompt_col (str): Name of the text-prompt column to keep.
        svg_col (str): Name of the SVG target column to keep.

    Returns:
        pd.DataFrame: Copy of ``df`` with missing values converted to strings, rows with
            empty prompts or SVGs removed, and the index reset.
    """
    out = df.copy()
    out[prompt_col] = out[prompt_col].fillna("").astype(str)
    out[svg_col] = out[svg_col].fillna("").astype(str)
    out = out[(out[prompt_col].str.strip() != "") & (out[svg_col].str.strip() != "")].copy()
    return out.reset_index(drop=True)


def format_for_seq2seq(
    df: pd.DataFrame,
    prompt_col: str,
    svg_col: str,
    prefix: str = "Generate SVG: ",
) -> pd.DataFrame:
    """Create seq2seq input and target text columns.

    Args:
        df (pd.DataFrame): Source DataFrame containing prompt and SVG data.
        prompt_col (str): Column holding the natural-language prompt text.
        svg_col (str): Column holding the target SVG string.
        prefix (str, optional): Instruction prefix prepended to each prompt before model
            tokenization. Defaults to ``"Generate SVG: "``.

    Returns:
        pd.DataFrame: Copy of ``df`` with ``input_text`` and ``target_text`` columns added
            and the index reset.
    """
    out = df.copy()
    out["input_text"] = prefix + out[prompt_col].fillna("").astype(str)
    out["target_text"] = out[svg_col].fillna("").astype(str)
    return out.reset_index(drop=True)


def select_easy_fraction(df: pd.DataFrame, easiest_frac: float = 0.20) -> pd.DataFrame:
    """Return the easiest slice of a difficulty-ranked DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame. If available, it should include either
            ``difficulty_percentile`` or ``final_difficulty_score`` so rows can be sorted
            from easy to hard.
        easiest_frac (float, optional): Fraction of rows to keep, expressed in ``[0, 1]``.
            At least one row is always returned. Defaults to ``0.20``.

    Returns:
        pd.DataFrame: The easiest ``easiest_frac`` portion of rows with a reset index.
    """
    out = df.copy()
    if "difficulty_percentile" in out.columns:
        out = out.sort_values("difficulty_percentile", ascending=True)
    elif "final_difficulty_score" in out.columns:
        out = out.sort_values("final_difficulty_score", ascending=True)
    n = max(1, int(len(out) * easiest_frac))
    return out.head(n).reset_index(drop=True)


def sample_n(df_in, n, seed=42):
    """Return all rows or a reproducible random sample.

    Args:
        df_in (pd.DataFrame): DataFrame to sample from.
        n (int | None): Number of rows to sample. If ``None`` or larger than the DataFrame,
            the full DataFrame is returned.
        seed (int, optional): Random seed passed to ``DataFrame.sample``. Defaults to ``42``.

    Returns:
        pd.DataFrame: Reset-index copy of the sampled rows.
    """
    if n is None or len(df_in) <= n:
        return df_in.copy().reset_index(drop=True)
    return df_in.sample(n=n, random_state=seed).reset_index(drop=True)


def train_val_split_df(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    seed: int = 42,
    *,
    shuffle: bool = True,
):
    """Split a DataFrame into train and validation subsets.

    Args:
        df (pd.DataFrame): Input examples to split.
        val_frac (float, optional): Fraction of rows assigned to the validation split.
            Defaults to ``0.10``.
        seed (int, optional): Random seed for ``train_test_split``. Defaults to ``42``.
        shuffle (bool, optional): If False, preserves input row order (sequential split).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(train_df, val_df)`` with reset indices.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=val_frac,
        random_state=seed,
        shuffle=shuffle,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_easy_subset(df_in, easy_frac):
    """Convenience wrapper that returns the easiest fraction of rows.

    Args:
        df_in (pd.DataFrame): Input DataFrame with optional difficulty columns.
        easy_frac (float): Fraction of easiest rows to return.

    Returns:
        pd.DataFrame: Reset-index copy of the selected easy subset.
    """
    subset = select_easy_fraction(df_in.copy(), easiest_frac=easy_frac)
    return subset.reset_index(drop=True).copy()


def get_hard_subset(df_in, easy_frac):
    """Return rows not included in the easy subset.

    Args:
        df_in (pd.DataFrame): Input DataFrame. Must contain a unique ``row_id`` column so
            easy rows can be excluded.
        easy_frac (float): Fraction used to define the easy subset.

    Returns:
        pd.DataFrame: Reset-index copy of the complementary hard subset.
    """
    easy_df = get_easy_subset(df_in, easy_frac)
    hard_df = df_in[~df_in["row_id"].isin(easy_df["row_id"])].copy()
    return hard_df.reset_index(drop=True)


def annotate_easy_hard(df_in, easy_frac=1 / 3):
    """Label each row as easy or hard based on a difficulty split.

    Args:
        df_in (pd.DataFrame): Input DataFrame. Must contain ``row_id`` and should contain a
            difficulty column compatible with ``select_easy_fraction``.
        easy_frac (float, optional): Fraction of rows labeled ``"easy"``. Remaining rows are
            labeled ``"hard"``. Defaults to ``1 / 3``.

    Returns:
        pd.DataFrame: Copy of ``df_in`` with a new ``difficulty_bucket`` string column.
    """
    easy_df = get_easy_subset(df_in, easy_frac)
    easy_ids = set(easy_df["row_id"].tolist())
    out = df_in.copy()
    out["difficulty_bucket"] = np.where(out["row_id"].isin(easy_ids), "easy", "hard")
    return out.reset_index(drop=True)


def sort_by_difficulty(df_in: pd.DataFrame) -> pd.DataFrame:
    """Sort rows from easy to hard using the best available difficulty column.

    Args:
        df_in (pd.DataFrame): Input DataFrame. If present, ``difficulty_percentile`` is used
            first, followed by ``final_difficulty_score``.

    Returns:
        pd.DataFrame: Sorted copy of ``df_in`` with a reset index. If no supported difficulty
            column exists, the original row order is preserved.
    """
    df = df_in.copy()
    if "difficulty_percentile" in df.columns:
        return df.sort_values("difficulty_percentile", ascending=True).reset_index(drop=True)
    if "final_difficulty_score" in df.columns:
        return df.sort_values("final_difficulty_score", ascending=True).reset_index(drop=True)
    return df.reset_index(drop=True)
