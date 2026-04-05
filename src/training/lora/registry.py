"""Append-only registry of trained LoRA adapters under outputs/models/."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

REGISTRY_FILENAME = "model_registry.csv"
ADAPTER_PREFIX = "lora_model_id_"


def new_model_id() -> str:
    """Return a new unique model id (no prefix)."""
    return uuid.uuid4().hex[:16]


def adapter_dir_name(model_id: str) -> str:
    return f"{ADAPTER_PREFIX}{model_id}"


def default_models_root(project_root: Path | str) -> Path:
    return Path(project_root) / "outputs" / "models"


def registry_csv_path(project_root: Path | str, *, models_root: Path | str | None = None) -> Path:
    root = Path(models_root) if models_root is not None else default_models_root(project_root)
    return root / REGISTRY_FILENAME


def register_model_from_adapter_dir(
    project_root: Path | str,
    source_adapter_dir: Path | str,
    *,
    curriculum: bool,
    tuning_stage: str,
    training_config: Mapping[str, Any],
    notes: str = "",
    model_id: str | None = None,
    models_root: Path | str | None = None,
) -> str:
    """Copy an existing adapter directory to ``<models_root>/lora_model_id_<id>/`` and append registry row.

    Args:
        project_root: Repository root (parent of ``outputs/``).
        source_adapter_dir: Directory containing saved PEFT adapter + tokenizer.
        curriculum: Whether curriculum training was used.
        tuning_stage: e.g. ``round1``, ``round2``, ``best_extended``.
        training_config: Serializable dict of hyperparameters and metadata.
        notes: Optional free text.
        model_id: Optional fixed id; if None, a new id is generated.
        models_root: If set, adapters and ``model_registry.csv`` live here (e.g.
            ``workflow_root / "models"``). Default: ``outputs/models``.

    Returns:
        The new ``model_id`` (without path prefix).
    """
    project_root = Path(project_root)
    source_adapter_dir = Path(source_adapter_dir)
    if not source_adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter dir not found: {source_adapter_dir}")

    mid = model_id or new_model_id()
    dest_root = Path(models_root) if models_root is not None else default_models_root(project_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / adapter_dir_name(mid)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(source_adapter_dir), str(dest))

    rel_adapter = str(dest.relative_to(project_root))
    append_registry_row(
        project_root,
        model_id=mid,
        adapter_dir=rel_adapter,
        curriculum=curriculum,
        tuning_stage=tuning_stage,
        training_config=dict(training_config),
        notes=notes,
        models_root=dest_root,
    )
    return mid


def append_registry_row(
    project_root: Path | str,
    *,
    model_id: str,
    adapter_dir: str,
    curriculum: bool,
    tuning_stage: str,
    training_config: dict[str, Any],
    notes: str = "",
    models_root: Path | str | None = None,
) -> None:
    """Append one row to ``model_registry.csv`` under ``models_root`` (default ``outputs/models``)."""
    project_root = Path(project_root)
    mroot = Path(models_root) if models_root is not None else default_models_root(project_root)
    mroot.mkdir(parents=True, exist_ok=True)
    csv_path = registry_csv_path(project_root, models_root=mroot)
    row = {
        "model_id": model_id,
        "adapter_dir": adapter_dir,
        "curriculum": bool(curriculum),
        "tuning_stage": tuning_stage,
        "training_config_json": json.dumps(training_config, sort_keys=True),
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def load_registry(project_root: Path | str, *, models_root: Path | str | None = None) -> pd.DataFrame:
    p = registry_csv_path(Path(project_root), models_root=models_root)
    if not p.is_file():
        return pd.DataFrame(
            columns=[
                "model_id",
                "adapter_dir",
                "curriculum",
                "tuning_stage",
                "training_config_json",
                "notes",
                "created_at",
            ]
        )
    return pd.read_csv(p)


def resolve_adapter_path(project_root: Path | str, adapter_dir: str) -> Path:
    """Resolve ``adapter_dir`` from registry (relative or absolute) to an existing path."""
    project_root = Path(project_root)
    p = Path(adapter_dir)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()
