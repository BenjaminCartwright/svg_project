"""Aggregate holdout prediction metrics and training metadata across registry models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.eval.holdout_evaluation import EVAL_RUN_MANIFEST_NAME, enrich_for_display
from src.training.lora.eval import summarize_generation_df
from src.training.lora.registry import load_registry

DEFAULT_CONFIG_KEYS = (
    "base_model_id",
    "max_seq_length",
    "learning_rate",
    "max_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
)


def default_eval_root_pairs(outputs_dir: Path | str) -> list[tuple[str, Path]]:
    ev = Path(outputs_dir) / "evaluations"
    return [
        ("holdout_tuning", ev / "holdout_tuning"),
        ("holdout_best_extended", ev / "holdout_best_extended"),
        ("holdout_curriculum", ev / "holdout_curriculum"),
    ]


def find_holdout_predictions_dir(
    model_id: str,
    postprocess_method: str,
    eval_roots: Sequence[tuple[str, Path]],
) -> tuple[str | None, Path | None]:
    """Return ``(eval_source_label, model_eval_dir)`` if postprocessed CSV exists."""
    fname = f"predictions_post_{postprocess_method}.csv"
    for label, root in eval_roots:
        d = Path(root) / str(model_id)
        if (d / fname).is_file():
            return label, d
    return None, None


def lookup_eval_loss_from_tuning_csvs(
    experiment_root: Path | str, registry_model_id: str
) -> float | None:
    """Last matching row in concatenated round1–round4 results (append order)."""
    experiment_root = Path(experiment_root)
    parts = []
    for name in (
        "round1_results.csv",
        "round2_results.csv",
        "round3_results.csv",
        "round4_results.csv",
    ):
        p = experiment_root / name
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        return None
    all_df = pd.concat(parts, ignore_index=True)
    if "registry_model_id" not in all_df.columns or "eval_loss" not in all_df.columns:
        return None
    sub = all_df[all_df["registry_model_id"].astype(str) == str(registry_model_id)]
    if len(sub) == 0:
        return None
    last = sub.iloc[-1]
    v = last["eval_loss"]
    if pd.isna(v):
        return None
    return float(v)


def _read_manifest(model_dir: Path) -> dict[str, Any] | None:
    p = model_dir / EVAL_RUN_MANIFEST_NAME
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_holdout_leaderboard_df(
    project_root: Path | str,
    outputs_dir: Path | str,
    model_ids: Sequence[str],
    postprocess_method: str,
    *,
    eval_roots: Sequence[tuple[str, Path]] | None = None,
    experiment_root: Path | str | None = None,
    workflow_root: Path | str | None = None,
) -> pd.DataFrame:
    """One row per ``model_id`` with registry fields, tuning ``eval_loss``, and holdout metrics.

    Predictions are loaded from ``<eval_root>/<model_id>/predictions_post_<method>.csv`` using
    the first eval root (in order) that contains that file.

    When ``workflow_root`` is set, eval roots, tuning CSV dir, and registry default to that
    profile tree (``workflow_root/evaluations/...``, ``workflow_root/lora_tuning_workflow``,
    ``workflow_root/models``). Otherwise ``outputs_dir`` is used as the layout root (legacy).
    """
    project_root = Path(project_root)
    outputs_dir = Path(outputs_dir)
    base = Path(workflow_root).resolve() if workflow_root is not None else outputs_dir
    roots = list(eval_roots) if eval_roots is not None else default_eval_root_pairs(base)
    exp_root = Path(experiment_root) if experiment_root is not None else base / "lora_tuning_workflow"
    models_root = base / "models"

    reg = load_registry(project_root, models_root=models_root)
    rows_out: list[dict[str, Any]] = []

    for mid in model_ids:
        mid = str(mid)
        row: dict[str, Any] = {"model_id": mid}
        rmatch = reg[reg["model_id"].astype(str) == mid]
        if len(rmatch) != 1:
            row["error"] = "model_id not found in registry (or duplicate)"
            rows_out.append(row)
            continue
        r = rmatch.iloc[0]
        row["tuning_stage"] = str(r.get("tuning_stage", ""))
        row["curriculum"] = r.get("curriculum")
        row["adapter_dir"] = str(r.get("adapter_dir", ""))

        try:
            cfg = json.loads(r["training_config_json"])
        except Exception:
            cfg = {}
        for k in DEFAULT_CONFIG_KEYS:
            row[f"cfg_{k}"] = cfg.get(k)

        row["eval_loss_tuning_csv"] = lookup_eval_loss_from_tuning_csvs(exp_root, mid)

        src_label, pred_dir = find_holdout_predictions_dir(mid, postprocess_method, roots)
        row["eval_predictions_source"] = src_label
        row["predictions_dir"] = str(pred_dir) if pred_dir else ""

        if pred_dir is None:
            row["error"] = (
                f"missing predictions_post_{postprocess_method}.csv "
                f"(searched: {[str(r[1]) for r in roots]})"
            )
            rows_out.append(row)
            continue

        man = _read_manifest(pred_dir)
        if man:
            row["manifest_max_new_tokens"] = man.get("max_new_tokens")
            row["manifest_base_model_id"] = man.get("base_model_id")

        try:
            pred_df = pd.read_csv(pred_dir / f"predictions_post_{postprocess_method}.csv")
        except Exception as e:
            row["error"] = f"failed to read predictions: {e}"
            rows_out.append(row)
            continue

        row["n_holdout_rows"] = len(pred_df)
        if "target_svg" in pred_df.columns:
            row["avg_target_char_len"] = float(
                pred_df["target_svg"].fillna("").astype(str).str.len().mean()
            )

        enriched = enrich_for_display(pred_df, "pred_svg")
        summ = summarize_generation_df(enriched)
        row.update(summ)
        if "path_count" in enriched.columns:
            row["avg_pred_path_count"] = float(enriched["path_count"].mean())

        rows_out.append(row)

    return pd.DataFrame(rows_out)
