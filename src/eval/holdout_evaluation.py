"""Holdout evaluation: raw generation, optional postprocess, CSV exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.core.modeling_splits import default_split_paths, load_split_manifest
from src.eval.postprocess_presets import get_postprocess_fn
from src.inference.generation import generate_svg_raw_prediction
from src.training.lora.eval import SVG_METRIC_COLUMN_NAMES, svg_metrics
from src.training.lora.modeling import load_inference_adapter
from src.training.prompts import format_svg_instruction_example

EVAL_RUN_MANIFEST_NAME = "eval_run_manifest.json"
EVAL_MANIFEST_SCHEMA_VERSION = 1


def holdout_eval_fingerprint(outputs_dir: Path | str) -> dict[str, Any]:
    """Stable fingerprint for the current holdout split (CSV bytes + split manifest subset)."""
    outputs_dir = Path(outputs_dir)
    paths = default_split_paths(outputs_dir)
    holdout_path = paths["holdout_eval"]
    if not holdout_path.is_file():
        raise FileNotFoundError(f"Missing holdout CSV: {holdout_path}")
    blob = holdout_path.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    ho = pd.read_csv(holdout_path)
    n = len(ho)
    manifest = load_split_manifest(outputs_dir)
    out: dict[str, Any] = {
        "holdout_csv_sha256": sha,
        "n_holdout_rows": int(n),
    }
    if manifest is not None:
        out["split_manifest_subset"] = {
            "seed": manifest.get("seed"),
            "holdout_n": manifest.get("holdout_n"),
            "source_csv": str(manifest.get("source_csv", "")),
            "first_n_labeled": manifest.get("first_n_labeled"),
        }
    return out


def build_eval_run_manifest(
    *,
    base_model_id: str,
    max_new_tokens: int,
    postprocess_method: str,
    adapter_dir_resolved: str,
    holdout_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVAL_MANIFEST_SCHEMA_VERSION,
        "base_model_id": str(base_model_id),
        "max_new_tokens": int(max_new_tokens),
        "postprocess_method": str(postprocess_method),
        "adapter_dir_resolved": str(adapter_dir_resolved),
        "holdout_fingerprint": dict(holdout_fingerprint),
    }


def _fingerprints_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_fingerprints_equal(a[k], b[k]) for k in sorted(a.keys()))
    return a == b


def raw_generation_manifest_matches(
    existing: dict[str, Any], requested: dict[str, Any]
) -> tuple[bool, str]:
    """Compare fields that affect raw GPU generation."""
    if int(existing.get("schema_version", 0)) != int(requested.get("schema_version", 0)):
        return False, "schema_version mismatch"
    for key in ("base_model_id", "max_new_tokens", "adapter_dir_resolved"):
        if existing.get(key) != requested.get(key):
            return False, f"{key} mismatch: {existing.get(key)!r} vs {requested.get(key)!r}"
    if not _fingerprints_equal(
        existing.get("holdout_fingerprint"), requested.get("holdout_fingerprint")
    ):
        return False, "holdout_fingerprint mismatch"
    return True, "ok"


def eval_run_manifest_matches(
    existing: dict[str, Any], requested: dict[str, Any]
) -> tuple[bool, str]:
    """Full match including postprocess method."""
    ok, msg = raw_generation_manifest_matches(existing, requested)
    if not ok:
        return ok, msg
    if existing.get("postprocess_method") != requested.get("postprocess_method"):
        return (
            False,
            f"postprocess_method mismatch: {existing.get('postprocess_method')!r} "
            f"vs {requested.get('postprocess_method')!r}",
        )
    return True, "ok"


def _read_eval_manifest(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / EVAL_RUN_MANIFEST_NAME
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_eval_run_manifest(out_dir: Path | str, manifest: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / EVAL_RUN_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _validate_cached_raw_df(
    raw_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    id_col: str,
) -> tuple[bool, str]:
    need = {id_col, "raw_pred", "target_svg", "prompt"}
    if not need.issubset(set(raw_df.columns)):
        return False, f"missing columns in cache: {need - set(raw_df.columns)}"
    if len(raw_df) != len(holdout_df):
        return False, f"row count {len(raw_df)} != holdout {len(holdout_df)}"
    a = set(raw_df[id_col].astype(str))
    b = set(holdout_df[id_col].astype(str))
    if a != b:
        return False, "holdout id set mismatch vs cached predictions"
    return True, "ok"


def load_holdout_predictions_cached_or_run(
    holdout_df: pd.DataFrame,
    prompt_col: str,
    svg_col: str,
    adapter_dir: Path | str,
    base_model_id: str,
    outputs_dir: Path | str,
    out_dir: Path | str,
    postprocess_method: str,
    *,
    max_new_tokens: int = 512,
    force_regenerate: bool = False,
    id_col: str = "id",
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load ``predictions_raw.csv`` when manifest matches; else generate and save.

    Args:
        show_progress: If True (default), show a tqdm bar while regenerating raw predictions.
        progress_desc: Optional tqdm description; defaults to ``\"Holdout SVG generation\"``.

    Returns:
        ``(dataframe_with_raw_pred_and_pred_svg, reason)`` where ``reason`` is
        ``cached_raw``, ``regenerated``, or ``postprocess_refreshed``.
    """
    out_dir = Path(out_dir)
    outputs_dir = Path(outputs_dir)
    adapter_resolved = str(Path(adapter_dir).resolve())
    fp = holdout_eval_fingerprint(outputs_dir)
    requested = build_eval_run_manifest(
        base_model_id=base_model_id,
        max_new_tokens=max_new_tokens,
        postprocess_method=postprocess_method,
        adapter_dir_resolved=adapter_resolved,
        holdout_fingerprint=fp,
    )

    raw_path = out_dir / "predictions_raw.csv"
    existing = _read_eval_manifest(out_dir)

    def _finalize_and_post(raw_df: pd.DataFrame) -> pd.DataFrame:
        raw_df = raw_df.copy()
        raw_df["pred_svg"] = apply_postprocess_column(raw_df, postprocess_method)
        post_path = out_dir / f"predictions_post_{postprocess_method}.csv"
        raw_df.to_csv(post_path, index=False)
        return raw_df

    if force_regenerate:
        raw_df = predict_holdout_raw(
            holdout_df,
            prompt_col,
            svg_col,
            adapter_dir,
            base_model_id,
            max_new_tokens=max_new_tokens,
            id_col=id_col,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        write_eval_run_manifest(out_dir, requested)
        out = _finalize_and_post(raw_df)
        return out, "regenerated"

    if raw_path.is_file() and existing is not None:
        ok_raw, _ = raw_generation_manifest_matches(existing, requested)
        if ok_raw:
            raw_df = pd.read_csv(raw_path)
            vok, _vmsg = _validate_cached_raw_df(raw_df, holdout_df, id_col)
            if vok:
                post_path = out_dir / f"predictions_post_{postprocess_method}.csv"
                if (
                    existing.get("postprocess_method") == postprocess_method
                    and post_path.is_file()
                ):
                    loaded = pd.read_csv(post_path)
                    if "pred_svg" in loaded.columns and len(loaded) == len(raw_df):
                        return loaded, "cached_raw"
                out = _finalize_and_post(raw_df)
                write_eval_run_manifest(out_dir, requested)
                return out, "postprocess_refreshed"

    raw_df = predict_holdout_raw(
        holdout_df,
        prompt_col,
        svg_col,
        adapter_dir,
        base_model_id,
        max_new_tokens=max_new_tokens,
        id_col=id_col,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    write_eval_run_manifest(out_dir, requested)
    out = _finalize_and_post(raw_df)
    return out, "regenerated"


def merge_ranked_metadata(holdout_df: pd.DataFrame, ranked_path: Path | str) -> pd.DataFrame:
    """Left-merge difficulty columns from ``train_ranked.csv`` on ``id``."""
    ranked_path = Path(ranked_path)
    if not ranked_path.is_file():
        return holdout_df.copy()
    ranked = pd.read_csv(ranked_path)
    if "id" not in ranked.columns or "id" not in holdout_df.columns:
        return holdout_df.copy()
    meta_cols = [
        c
        for c in ("difficulty_percentile", "final_difficulty_score", "difficulty_bucket")
        if c in ranked.columns
    ]
    if not meta_cols:
        return holdout_df.copy()
    sub = ranked[["id"] + meta_cols].drop_duplicates(subset=["id"])
    return holdout_df.merge(sub, on="id", how="left")


def build_full_prompts(
    df: pd.DataFrame,
    prompt_col: str,
) -> pd.Series:
    return df[prompt_col].fillna("").astype(str).apply(
        lambda p: format_svg_instruction_example(p, svg_text=None, include_answer=False)
    )


def predict_holdout_raw(
    holdout_df: pd.DataFrame,
    prompt_col: str,
    svg_col: str,
    adapter_dir: Path | str,
    base_model_id: str,
    *,
    max_new_tokens: int = 512,
    id_col: str = "id",
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    """Generate raw continuations for each holdout row.

    Args:
        show_progress: If True (default), show a tqdm bar over holdout rows.
        progress_desc: tqdm description; default ``\"Holdout SVG generation\"``.
    """
    tokenizer, model = load_inference_adapter(adapter_dir, base_model_id)
    full_prompts = build_full_prompts(holdout_df, prompt_col)
    rows = []
    desc = progress_desc if progress_desc is not None else "Holdout SVG generation"
    row_iter = holdout_df.iterrows()
    if show_progress:
        row_iter = tqdm(row_iter, total=len(holdout_df), desc=desc)
    for i, row in row_iter:
        fp = full_prompts.loc[i]
        raw = generate_svg_raw_prediction(fp, tokenizer, model, max_new_tokens=max_new_tokens)
        rid = row[id_col]
        rows.append(
            {
                id_col: rid,
                "prompt": str(row[prompt_col]),
                "target_svg": str(row.get(svg_col, "") or ""),
                "full_prompt": fp,
                "raw_pred": raw,
            }
        )
    return pd.DataFrame(rows)


def apply_postprocess_column(raw_df: pd.DataFrame, method_name: str, raw_col: str = "raw_pred") -> pd.Series:
    fn = get_postprocess_fn(method_name)
    return raw_df[raw_col].fillna("").astype(str).map(fn)


def score_predictions_df(pred_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Append svg_metrics columns for each row."""
    metrics_list = []
    for _, row in pred_df.iterrows():
        m = svg_metrics(row[pred_col])
        metrics_list.append(m)
    met_df = pd.DataFrame(metrics_list)
    return pd.concat([pred_df.reset_index(drop=True), met_df], axis=1)


def enrich_for_display(pred_df: pd.DataFrame, pred_col: str = "pred_svg") -> pd.DataFrame:
    """Ensure ``pred_svg`` and metric columns exist for ``display_*`` / ``summarize_generation_df``."""
    out = pred_df.copy()
    drop_metrics = [c for c in SVG_METRIC_COLUMN_NAMES if c in out.columns]
    if drop_metrics:
        out = out.drop(columns=drop_metrics)
    if pred_col != "pred_svg":
        out["pred_svg"] = out[pred_col].fillna("").astype(str)
    else:
        out["pred_svg"] = out["pred_svg"].fillna("").astype(str)
    return score_predictions_df(out, "pred_svg")


def save_evaluation_bundle(
    out_dir: Path | str,
    pred_df: pd.DataFrame,
    *,
    postprocess_name: str | None = None,
) -> Path:
    """Write predictions CSV; optional second file with postprocessed column."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "predictions_raw.csv"
    pred_df.to_csv(raw_path, index=False)
    if postprocess_name and "pred_svg" in pred_df.columns:
        post_path = out_dir / f"predictions_post_{postprocess_name}.csv"
        pred_df.to_csv(post_path, index=False)
    return raw_path


def sample_percentile_buckets(
    df: pd.DataFrame,
    buckets: Sequence[tuple[str, float, float]],
    n_per_bucket: int,
    seed: int,
    percentile_col: str = "difficulty_percentile",
) -> pd.DataFrame:
    """Sample up to ``n_per_bucket`` rows from each percentile band (like notebook 09)."""
    if percentile_col not in df.columns:
        n = min(n_per_bucket * max(1, len(buckets)), len(df))
        out = df.sample(n=min(n, len(df)), random_state=seed).copy()
        out["difficulty_bucket_label"] = "all"
        return out.reset_index(drop=True)
    parts = []
    for bucket_label, low, high in buckets:
        if high >= 1.0:
            bucket_df = df[(df[percentile_col] >= low) & (df[percentile_col] <= high)].copy()
        else:
            bucket_df = df[(df[percentile_col] >= low) & (df[percentile_col] < high)].copy()
        if len(bucket_df) == 0:
            continue
        n_draw = min(n_per_bucket, len(bucket_df))
        sampled = bucket_df.sample(n=n_draw, random_state=seed).reset_index(drop=True)
        sampled["difficulty_bucket_label"] = bucket_label
        parts.append(sampled)
    if not parts:
        raise ValueError("No rows available for the configured percentile buckets.")
    return pd.concat(parts, ignore_index=True)
