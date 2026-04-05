"""Holdout split utilities for modeling: train/val pool vs evaluation holdout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.dataframe import choose_first_existing, select_easy_fraction
from src.svg.cleaning import clean_svg
from src.training.seq2seq.preprocess import prepare_seq2seq_dataframe


def make_holdout_split(
    df: pd.DataFrame,
    holdout_n: int,
    seed: int = 42,
    id_col: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into a training/validation pool and a fixed holdout set.

    Args:
        df: Full labeled training table (must include ``id_col``).
        holdout_n: Number of rows reserved for holdout evaluation (capped by len(df)).
        seed: Random seed for reproducibility.
        id_col: Column name for stable example identifiers.

    Returns:
        Tuple ``(train_val_pool, holdout_df)``, both reset-index copies.
    """
    if id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not found. Columns: {list(df.columns)}")
    work = df.copy().reset_index(drop=True)
    n = min(int(holdout_n), len(work))
    if n <= 0:
        raise ValueError("holdout_n must be positive")
    if n >= len(work):
        raise ValueError("holdout_n must be smaller than dataframe length")
    # test_size=int -> exactly n rows in the test split (holdout)
    pool_df, holdout_df = train_test_split(
        work,
        test_size=n,
        random_state=seed,
        shuffle=True,
    )
    return pool_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def default_split_paths(outputs_dir: Path | str) -> dict[str, Path]:
    """Paths under ``<outputs_dir>/modeling_splits/``.

    For workflow-scoped runs, pass ``workflow_root`` (e.g.
    ``outputs/workflow_runs/<RUN_PROFILE_ID>``) so splits live under that profile.
    """
    base = Path(outputs_dir) / "modeling_splits"
    return {
        "dir": base,
        "holdout_eval": base / "holdout_eval.csv",
        "train_val_pool": base / "train_val_pool.csv",
        "manifest": base / "split_manifest.json",
    }


def split_paths_for_workflow_root(workflow_root: Path | str) -> dict[str, Path]:
    """Same as ``default_split_paths(workflow_root)`` — alias for readability."""
    return default_split_paths(workflow_root)


def save_split_artifacts(
    train_val_pool: pd.DataFrame,
    holdout_df: pd.DataFrame,
    outputs_dir: Path | str,
    *,
    seed: int,
    holdout_n: int,
    source_csv: str,
    prompt_col: str,
    svg_col: str,
    first_n_labeled: int | None = None,
    run_profile_id: str = "default",
    use_easy_subset: bool = False,
    easy_subset_frac: float | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write pool, holdout CSVs and a small JSON manifest."""
    paths = default_split_paths(outputs_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    train_val_pool.to_csv(paths["train_val_pool"], index=False)
    holdout_df.to_csv(paths["holdout_eval"], index=False)
    manifest: dict[str, Any] = {
        "seed": seed,
        "holdout_n": holdout_n,
        "source_csv": str(source_csv),
        "prompt_col": prompt_col,
        "svg_col": svg_col,
        "first_n_labeled": first_n_labeled,
        "run_profile_id": str(run_profile_id),
        "use_easy_subset": bool(use_easy_subset),
        "easy_subset_frac": easy_subset_frac,
        "train_val_pool_rows": len(train_val_pool),
        "holdout_rows": len(holdout_df),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def load_train_val_pool(outputs_dir: Path | str) -> pd.DataFrame:
    p = default_split_paths(outputs_dir)["train_val_pool"]
    if not p.is_file():
        raise FileNotFoundError(f"Missing train/val pool split: {p}")
    return pd.read_csv(p)


def load_holdout_eval(outputs_dir: Path | str) -> pd.DataFrame:
    p = default_split_paths(outputs_dir)["holdout_eval"]
    if not p.is_file():
        raise FileNotFoundError(f"Missing holdout split: {p}")
    return pd.read_csv(p)


def load_split_manifest(outputs_dir: Path | str) -> dict[str, Any] | None:
    """Return parsed ``split_manifest.json`` if it exists, else ``None``."""
    p = default_split_paths(outputs_dir)["manifest"]
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def split_artifacts_exist(outputs_dir: Path | str) -> bool:
    """True if both pool and holdout CSVs exist."""
    paths = default_split_paths(outputs_dir)
    return paths["train_val_pool"].is_file() and paths["holdout_eval"].is_file()


def load_existing_split_tables(outputs_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/val pool and holdout from disk (both files must exist)."""
    paths = default_split_paths(outputs_dir)
    if not split_artifacts_exist(outputs_dir):
        raise FileNotFoundError(
            f"Missing split CSVs under {paths['dir']}. Run a fresh split or set FORCE_REBUILD_SPLITS."
        )
    return pd.read_csv(paths["train_val_pool"]), pd.read_csv(paths["holdout_eval"])


def manifest_matches_params(
    manifest: dict[str, Any] | None,
    *,
    seed: int,
    holdout_n: int,
    source_csv: str,
    first_n_labeled: int | None = None,
    run_profile_id: str = "default",
    use_easy_subset: bool = False,
    easy_subset_frac: float | None = None,
) -> tuple[bool, str]:
    """Check saved manifest matches requested split parameters.

    Returns:
        ``(ok, message)`` where ``ok`` is True if safe to reuse, or manifest is missing.
    """
    if manifest is None:
        return True, "no manifest (CSV files only); reusing splits"
    m_prof = str(manifest.get("run_profile_id", "default"))
    if m_prof != str(run_profile_id):
        return (
            False,
            f"manifest run_profile_id {m_prof!r} != requested {run_profile_id!r}",
        )
    m_easy = bool(manifest.get("use_easy_subset", False))
    if m_easy != bool(use_easy_subset):
        return False, f"manifest use_easy_subset {m_easy} != requested {use_easy_subset}"
    m_ef = manifest.get("easy_subset_frac")
    req_ef = easy_subset_frac
    if use_easy_subset:
        if m_ef is None or req_ef is None:
            if m_ef != req_ef:
                return False, f"manifest easy_subset_frac {m_ef} != requested {req_ef}"
        elif float(m_ef) != float(req_ef):
            return False, f"manifest easy_subset_frac {m_ef} != requested {req_ef}"
    m_seed = manifest.get("seed")
    m_n = manifest.get("holdout_n")
    m_src = str(manifest.get("source_csv", ""))
    src = str(source_csv)
    if m_seed is not None and int(m_seed) != int(seed):
        return False, f"manifest seed {m_seed} != requested SEED {seed}"
    if m_n is not None and int(m_n) != int(holdout_n):
        return False, f"manifest holdout_n {m_n} != requested HOLDOUT_N {holdout_n}"
    if m_src and src and m_src != src:
        return False, f"manifest source_csv {m_src!r} != current data_path {src!r}"
    m_fn = manifest.get("first_n_labeled")
    m_first_n = None if m_fn is None else int(m_fn)
    if first_n_labeled is None:
        if m_first_n is not None:
            return (
                False,
                f"manifest first_n_labeled {m_first_n} != requested None (use all rows)",
            )
    elif m_first_n != int(first_n_labeled):
        return (
            False,
            f"manifest first_n_labeled {m_first_n} != requested FIRST_N_LABELED {first_n_labeled}",
        )
    return True, "manifest matches parameters"


def build_pool_and_holdout(
    *,
    data_path: Path | str,
    workflow_root: Path | str,
    seed: int,
    holdout_n: int,
    run_profile_id: str,
    use_easy_subset: bool,
    easy_subset_frac: float,
    first_n_labeled: int | None,
    ranked_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labeled data, clean, optionally easy-subset, split, and persist pool + holdout.

    Mirrors notebook **03** so split logic stays out of notebooks.
    """
    data_path = Path(data_path)
    ranked_path = Path(ranked_path)
    df = pd.read_csv(data_path)
    print("Loaded:", data_path, df.shape)
    prompt_col = choose_first_existing(df, ["prompt", "description", "text"], "df")
    svg_col = choose_first_existing(df, ["svg", "svg_code", "target", "label"], "df")
    working_df = prepare_seq2seq_dataframe(df.copy(), prompt_col=prompt_col, svg_col=svg_col)
    working_df[svg_col] = working_df[svg_col].astype(str).apply(clean_svg)
    working_df = working_df.reset_index(drop=True)
    if "id" not in working_df.columns:
        working_df["id"] = working_df.index.astype(str)
    has_diff = "difficulty_percentile" in working_df.columns or "final_difficulty_score" in working_df.columns
    if use_easy_subset and not has_diff:
        if not ranked_path.is_file():
            raise ValueError(
                "USE_EASY_SUBSET needs difficulty columns or train_ranked.csv (notebook 02). "
                "Set USE_EASY_SUBSET = False or use ranked data."
            )
        ranked = pd.read_csv(ranked_path)
        id_set = set(working_df["id"].astype(str))
        sub = ranked[ranked["id"].astype(str).isin(id_set)]
        merge_cols = ["id"] + [
            c
            for c in ("difficulty_percentile", "final_difficulty_score", "difficulty_bucket")
            if c in sub.columns
        ]
        sub = sub[merge_cols].drop_duplicates(subset=["id"], keep="first")
        working_df = working_df.merge(sub, on="id", how="left")
        has_diff = "difficulty_percentile" in working_df.columns or "final_difficulty_score" in working_df.columns
        if not has_diff:
            raise ValueError("After merging train_ranked.csv, still no difficulty columns for easy subset.")
    if use_easy_subset:
        working_df = select_easy_fraction(working_df, easiest_frac=float(easy_subset_frac))
        print("Easy subset frac", easy_subset_frac, "-> rows", len(working_df))
    if first_n_labeled is not None:
        working_df = working_df.iloc[: int(first_n_labeled)].copy().reset_index(drop=True)
        print("Using first", first_n_labeled, "rows after clean / subset:", working_df.shape)
    pool, hold = make_holdout_split(working_df, holdout_n, seed=seed, id_col="id")
    paths = save_split_artifacts(
        pool,
        hold,
        workflow_root,
        seed=seed,
        holdout_n=holdout_n,
        source_csv=str(data_path),
        prompt_col=prompt_col,
        svg_col=svg_col,
        first_n_labeled=first_n_labeled,
        run_profile_id=str(run_profile_id),
        use_easy_subset=bool(use_easy_subset),
        easy_subset_frac=float(easy_subset_frac) if use_easy_subset else None,
    )
    print("Saved:", paths["train_val_pool"])
    print("Saved:", paths["holdout_eval"])
    return pool, hold
