"""Per-workflow layout manifest: paths to splits, models, tuning, eval, and prediction CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAYOUT_FILENAME = "workflow_layout.json"


def _read_layout(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_layout(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_workflow_layout_stub(
    project_root: Path | str,
    workflow_root: Path | str,
    *,
    run_profile_id: str,
    split_manifest: dict[str, Any] | None = None,
) -> Path:
    """Create or overwrite ``workflow_layout.json`` with standard relative paths."""
    project_root = Path(project_root).resolve()
    workflow_root = Path(workflow_root).resolve()
    layout_path = workflow_root / LAYOUT_FILENAME

    def rel(p: Path) -> str:
        return str(p.resolve().relative_to(project_root))

    ev = workflow_root / "evaluations"
    data: dict[str, Any] = {
        "schema_version": 1,
        "run_profile_id": str(run_profile_id),
        "workflow_root": rel(workflow_root),
        "modeling_splits_dir": rel(workflow_root / "modeling_splits"),
        "models_root": rel(workflow_root / "models"),
        "model_registry_csv": rel(workflow_root / "models" / "model_registry.csv"),
        "lora_tuning_workflow_dir": rel(workflow_root / "lora_tuning_workflow"),
        "evaluations": {
            "holdout_tuning": rel(ev / "holdout_tuning"),
            "holdout_best_extended": rel(ev / "holdout_best_extended"),
            "holdout_curriculum": rel(ev / "holdout_curriculum"),
        },
        "predictions_by_model_id": {},
    }
    if split_manifest:
        data["split_manifest_snapshot"] = split_manifest
    _write_layout(layout_path, data)
    return layout_path


def update_workflow_layout_prediction(
    project_root: Path | str,
    workflow_root: Path | str,
    model_id: str,
    predictions_csv: Path | str,
    *,
    eval_kind: str = "holdout_tuning",
) -> Path:
    """Record relative path to a model's holdout predictions CSV (postprocessed or raw)."""
    project_root = Path(project_root).resolve()
    workflow_root = Path(workflow_root).resolve()
    layout_path = workflow_root / LAYOUT_FILENAME
    data = _read_layout(layout_path)
    if not data:
        write_workflow_layout_stub(
            project_root, workflow_root, run_profile_id=workflow_root.name
        )
        data = _read_layout(layout_path)
    pred = Path(predictions_csv).resolve()
    rel_pred = str(pred.relative_to(project_root))
    by_mid = data.setdefault("predictions_by_model_id", {})
    by_mid[str(model_id)] = {
        "predictions_csv": rel_pred,
        "eval_kind": eval_kind,
    }
    _write_layout(layout_path, data)
    return layout_path
