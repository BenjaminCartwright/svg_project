from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm.auto import tqdm

from src.core.dataframe import choose_first_existing
from src.inference.generation import generate_svg_prediction
from src.inference.postprocess import sanitize_svg_prediction
from src.svg.cleaning import validate_svg_constraints
from src.training.lora.modeling import load_inference_adapter


def load_kaggle_inputs(
    test_csv_path,
    adapter_dir,
    model_id,
    prompt_candidates=None,
):
    """Load and validate submission-time input artifacts.

    Args:
        test_csv_path (str | pathlib.Path): Path to the Kaggle-style test CSV.
        adapter_dir (str | pathlib.Path): Directory containing the saved LoRA adapter files.
        model_id (str): Base model identifier used later when reloading the adapter.
        prompt_candidates (list[str] | None, optional): Candidate prompt column names to try in
            order. Defaults to ``["prompt", "description", "text"]``.

    Returns:
        tuple[pd.DataFrame, str, dict[str, pathlib.Path | str]]: ``(test_df, prompt_col,
        metadata)`` where ``metadata`` records the resolved paths and ``model_id``.

    Raises:
        FileNotFoundError: If the test CSV or adapter directory is missing.
        ValueError: If the CSV does not contain an ``id`` column or any candidate prompt column.
    """
    test_csv_path = Path(test_csv_path)
    adapter_dir = Path(adapter_dir)
    if prompt_candidates is None:
        prompt_candidates = ["prompt", "description", "text"]

    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Missing test CSV: {test_csv_path}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    test_df = pd.read_csv(test_csv_path)
    if "id" not in test_df.columns:
        raise ValueError(f"`id` column missing from test CSV. Columns were: {list(test_df.columns)}")
    prompt_col = choose_first_existing(test_df, prompt_candidates, "test_df")
    return test_df, prompt_col, {"test_csv_path": test_csv_path, "adapter_dir": adapter_dir, "model_id": model_id}


def load_submission_model(adapter_dir, model_id):
    """Load the tokenizer and PEFT model used for submission inference.

    Args:
        adapter_dir (str | pathlib.Path): Directory containing the saved adapter weights.
        model_id (str): Base model identifier compatible with the adapter.

    Returns:
        tuple[Any, Any]: ``(tokenizer, model)`` loaded via ``load_inference_adapter`` with the
            model placed in evaluation mode.
    """
    tokenizer, model = load_inference_adapter(adapter_dir, model_id)
    model.eval()
    return tokenizer, model


def predict_one_svg(
    prompt,
    tokenizer,
    model,
    max_new_tokens=768,
):
    """Generate one sanitized SVG prediction for a prompt.

    Args:
        prompt (Any): Prompt value describing the desired SVG. It is coerced to ``str``.
        tokenizer: Tokenizer used for autoregressive generation.
        model: Loaded inference model.
        max_new_tokens (int, optional): Maximum generated continuation length. Defaults to ``768``.

    Returns:
        str: Sanitized SVG string ready for validation or submission.
    """
    raw_svg = generate_svg_prediction(
        prompt_text=str(prompt),
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )
    return sanitize_svg_prediction(raw_svg)


def validate_submission_svg(svg_text):
    """Validate a single SVG using the project's submission constraints.

    Args:
        svg_text (str): SVG string to validate.

    Returns:
        dict[str, bool | int]: Constraint metrics from ``validate_svg_constraints`` including the
            aggregate ``is_valid_submission_svg`` flag.
    """
    return validate_svg_constraints(svg_text)


def build_submission_rows(
    test_df,
    prompt_col,
    tokenizer,
    model,
    max_new_tokens=768,
    show_progress=True,
    progress_desc="Generating SVGs",
):
    """Generate submission rows for every example in a test DataFrame.

    Args:
        test_df (pd.DataFrame): Test-set DataFrame containing at least ``id`` and ``prompt_col``.
        prompt_col (str): Name of the prompt column to read from each row.
        tokenizer: Tokenizer used for generation.
        model: Loaded inference model.
        max_new_tokens (int, optional): Maximum tokens generated per row. Defaults to ``768``.
        show_progress (bool, optional): If ``True`` (default), show a tqdm progress bar.
        progress_desc (str, optional): tqdm description string. Defaults to ``"Generating SVGs"``.

    Returns:
        pd.DataFrame: Submission DataFrame with exactly ``id`` and ``svg`` columns. The elapsed
        generation time in seconds is also stored in ``submission_df.attrs["elapsed_seconds"]``.
    """
    submission_rows = []
    started_at = perf_counter()
    total_rows = len(test_df)

    row_iter = test_df.itertuples(index=False)
    if show_progress:
        row_iter = tqdm(row_iter, total=total_rows, desc=progress_desc)

    for row in row_iter:
        row_id = getattr(row, "id")
        prompt = getattr(row, prompt_col)
        svg_text = predict_one_svg(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
        )
        submission_rows.append({"id": row_id, "svg": svg_text})

    elapsed_s = perf_counter() - started_at
    submission_df = pd.DataFrame(submission_rows, columns=["id", "svg"])
    submission_df.attrs["elapsed_seconds"] = elapsed_s
    return submission_df


def assert_submission_ready(submission_df, validation_df, expected_rows=None):
    """Assert that submission and validation DataFrames are internally consistent.

    Args:
        submission_df (pd.DataFrame): Submission DataFrame expected to contain exactly ``id`` and
            ``svg`` columns.
        validation_df (pd.DataFrame): Validation DataFrame containing one row per submission row
            and an ``is_valid_submission_svg`` boolean column.
        expected_rows (int | None, optional): Expected submission length, if known. Defaults to
            ``None``.

    Returns:
        bool: ``True`` when all checks pass.

    Raises:
        ValueError: If required columns are missing, row counts mismatch, IDs are invalid, SVGs
            are empty, or any validation row is marked invalid.
    """
    if list(submission_df.columns) != ["id", "svg"]:
        raise ValueError(f"Submission columns must be exactly ['id', 'svg']; got {list(submission_df.columns)}")
    if expected_rows is not None and len(submission_df) != int(expected_rows):
        raise ValueError(f"Submission has {len(submission_df)} rows but expected {expected_rows}")
    if submission_df["id"].isna().any():
        raise ValueError("Submission contains missing ids")
    if not submission_df["id"].is_unique:
        raise ValueError("Submission ids must be unique")
    if submission_df["svg"].isna().any():
        raise ValueError("Submission contains missing SVG values")
    if (submission_df["svg"].astype(str).str.strip() == "").any():
        raise ValueError("Submission contains empty SVG values")
    if len(validation_df) != len(submission_df):
        raise ValueError("Validation dataframe row count does not match submission dataframe")
    if "is_valid_submission_svg" not in validation_df.columns:
        raise ValueError("Validation dataframe is missing `is_valid_submission_svg`")

    invalid_df = validation_df.loc[~validation_df["is_valid_submission_svg"]]
    if len(invalid_df):
        invalid_ids = invalid_df["id"].astype(str).head(10).tolist()
        raise ValueError(
            f"Submission contains {len(invalid_df)} invalid SVG rows. Example ids: {invalid_ids}"
        )
    return True


def write_submission_csv(submission_df, output_path="/kaggle/working/submission.csv"):
    """Write the final submission DataFrame to disk.

    Args:
        submission_df (pd.DataFrame): DataFrame with ``id`` and ``svg`` columns.
        output_path (str | pathlib.Path, optional): Output CSV path. Parent directories are
            created if needed. Defaults to ``"/kaggle/working/submission.csv"``.

    Returns:
        pathlib.Path: Resolved output path used for writing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path


def validate_submission_csv(output_path, expected_rows=None):
    """Reload a submission CSV and validate every row.

    Args:
        output_path (str | pathlib.Path): Path to a submission CSV with ``id`` and ``svg``
            columns.
        expected_rows (int | None, optional): Expected number of rows, if known. Defaults to
            ``None``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(submission_df, validation_df)`` where
            ``validation_df`` contains ``id`` and ``is_valid_submission_svg``.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If the CSV schema or row contents are invalid.
    """
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Submission CSV not found: {output_path}")

    submission_df = pd.read_csv(output_path)
    if list(submission_df.columns) != ["id", "svg"]:
        raise ValueError(
            f"Submission CSV columns must be exactly ['id', 'svg']; got {list(submission_df.columns)}"
        )
    if expected_rows is not None and len(submission_df) != int(expected_rows):
        raise ValueError(f"Submission CSV has {len(submission_df)} rows but expected {expected_rows}")
    if submission_df["id"].isna().any():
        raise ValueError("Submission CSV contains missing ids")
    if not submission_df["id"].is_unique:
        raise ValueError("Submission CSV ids must be unique")
    if submission_df["svg"].isna().any():
        raise ValueError("Submission CSV contains missing SVG values")
    if (submission_df["svg"].astype(str).str.strip() == "").any():
        raise ValueError("Submission CSV contains empty SVG values")

    validation_df = pd.DataFrame(
        {
            "id": submission_df["id"],
            "is_valid_submission_svg": submission_df["svg"].map(
                lambda svg_text: validate_submission_svg(svg_text)["is_valid_submission_svg"]
            ),
        }
    )
    invalid_df = validation_df.loc[~validation_df["is_valid_submission_svg"]]
    if len(invalid_df):
        invalid_ids = invalid_df["id"].astype(str).head(10).tolist()
        raise ValueError(
            f"Submission CSV contains {len(invalid_df)} invalid SVG rows. Example ids: {invalid_ids}"
        )
    return submission_df, validation_df


def sample_validation_report(validation_df, n=10):
    """Return a small report of failed validation rows.

    Args:
        validation_df (pd.DataFrame): Validation DataFrame containing
            ``is_valid_submission_svg``.
        n (int, optional): Maximum number of failed rows to return. Defaults to ``10``.

    Returns:
        pd.DataFrame: Reset-index DataFrame containing up to ``n`` invalid rows.
    """
    if "is_valid_submission_svg" not in validation_df.columns:
        raise ValueError("Validation dataframe is missing `is_valid_submission_svg`")
    failed_df = validation_df.loc[~validation_df["is_valid_submission_svg"]].head(int(n)).copy()
    return failed_df.reset_index(drop=True)


def preview_predictions(submission_df, n=3):
    """Return the first few generated submission rows.

    Args:
        submission_df (pd.DataFrame): Submission DataFrame with generated predictions.
        n (int, optional): Number of rows to preview. Defaults to ``3``.

    Returns:
        pd.DataFrame: Reset-index copy of the first ``n`` rows.
    """
    return submission_df.head(int(n)).copy().reset_index(drop=True)


def timed_inference_summary(submission_df):
    """Summarize throughput from a submission DataFrame.

    Args:
        submission_df (pd.DataFrame): Submission DataFrame whose ``attrs`` may include
            ``elapsed_seconds``.

    Returns:
        dict[str, int | float]: Row count, elapsed seconds, and rows-per-second throughput.
    """
    elapsed_s = float(submission_df.attrs.get("elapsed_seconds", 0.0))
    row_count = int(len(submission_df))
    rows_per_second = (row_count / elapsed_s) if elapsed_s > 0 else 0.0
    return {
        "rows": row_count,
        "elapsed_seconds": elapsed_s,
        "rows_per_second": rows_per_second,
    }


__all__ = [
    "assert_submission_ready",
    "build_submission_rows",
    "load_kaggle_inputs",
    "load_submission_model",
    "predict_one_svg",
    "preview_predictions",
    "sample_validation_report",
    "timed_inference_summary",
    "validate_submission_csv",
    "validate_submission_svg",
    "write_submission_csv",
]
