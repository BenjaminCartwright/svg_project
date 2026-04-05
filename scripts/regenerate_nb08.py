"""Regenerate notebooks/08_eval_holdout_all_tuning_models.ipynb (per-model cells)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

MAX_SLOTS_DEFAULT = 16


def generate_nb08(max_slots: int = MAX_SLOTS_DEFAULT, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    nb_path = root / "notebooks" / "08_eval_holdout_all_tuning_models.ipynb"

    def md(s):
        return nbf.v4.new_markdown_cell(s)

    def code(s):
        return nbf.v4.new_code_cell(s)

    cells = [
        md(
            """# 08 — Holdout evaluation: all tuning models (rounds 1–4)

**Summary:** Evaluate every registry row with `tuning_stage` in `round1`–`round4` on the **holdout** set from notebook **03** (labeled `id` + target SVG). Predictions are **raw** decode; postprocessing for display/submission uses **`POSTPROCESS_METHOD`** (`src.eval.postprocess_presets`).

**Recommended run order:** 03 → 04 → 05 → **08** (this notebook) → 06 → 07 → **09** (post-training holdout eval) → 10–12.

**Workflow (same idea as legacy notebook 09):**
1. Run setup cells to build `HOLDOUT_MODELS` and `_INFER_COMMON`.
2. **Either** run **one cell** to generate for **all** models, **or** run **individual generation cells** per model slot (skips automatically if that slot has no model).
3. Run **display cells** for each model index you generated (after its generation cell).

**Disk cache:** Each model folder under `EVAL_ROOT/<model_id>/` stores `predictions_raw.csv`, `predictions_post_<method>.csv`, and `eval_run_manifest.json`. If you re-run with the same holdout split, base model, adapter path, `MAX_NEW_TOKENS`, and `POSTPROCESS_METHOD`, predictions are **reloaded** from disk (no GPU regen). Set `FORCE_REGENERATE_HOLDOUT_PREDICTIONS = True` to ignore the manifest.

**Profile:** Set `RUN_PROFILE_ID` to match notebook **03**; splits, registry, and evals live under `outputs/workflow_runs/<RUN_PROFILE_ID>/`.

**Parameters:** Adjust `RUN_PROFILE_ID`, `MODEL_IDS` (empty = all round1–4 rows; else filter, **in list order**), `POSTPROCESS_METHOD`, `FORCE_REGENERATE_HOLDOUT_PREDICTIONS`, bucket sampling, and `MAX_MODEL_SLOTS` (must be ≥ `len(HOLDOUT_MODELS)`; extra slots no-op)."""
        ),
        md("#### Colab / install"),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "!pip -q install transformers peft accelerate bitsandbytes cairosvg pillow matplotlib lxml pandas tqdm"
        ),
        md("#### Imports, paths, and parameters"),
        code(
            f"""import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
EVAL_ROOT = WORKFLOW_ROOT / 'evaluations' / 'holdout_tuning'

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
MODEL_IDS = []  # empty: all tuning-stage models; else subset in this order
MAX_NEW_TOKENS = 512
POSTPROCESS_METHOD = 'current_default_sanitizer'

SEED = 42
USE_PERCENTILE_BUCKETS = True
PERCENTILE_BUCKETS = [
    ('0-10 percentile', 0.00, 0.10),
    ('40-60 percentile', 0.40, 0.60),
    ('80-100 percentile', 0.80, 1.00),
]
N_SAMPLES_PER_BUCKET = 5
N_TEXT_ROWS = 12
N_RENDER_ROWS = 6
PREVIEW_CHARS = 8000
SHOW_RAW_METRICS = True
# True: always run raw generation; False: reuse disk cache when eval_run_manifest.json matches
FORCE_REGENERATE_HOLDOUT_PREDICTIONS = False

# Number of per-model cell pairs below; increase if you have more registry rows than slots
MAX_MODEL_SLOTS = {max_slots}

print('cuda:', torch.cuda.is_available())
print('POSTPROCESS_METHOD:', POSTPROCESS_METHOD)
print('FORCE_REGENERATE_HOLDOUT_PREDICTIONS:', FORCE_REGENERATE_HOLDOUT_PREDICTIONS)
print('MAX_MODEL_SLOTS:', MAX_MODEL_SLOTS)"""
        ),
        md(
            """#### Discover registry models and build session

Builds `HOLDOUT_MODELS` (ordered list), `SESSION` (in-memory display cache), and `_INFER_COMMON` kwargs shared by batch and per-slot generation. On-disk prediction reuse uses `eval_run_manifest.json` (see summary above)."""
        ),
        code(
            """from src.eval.holdout_tuning_notebook import (
    HoldoutTuningSession,
    build_holdout_models_list,
    prepare_holdout_and_columns,
)
from src.eval.postprocess_presets import POSTPROCESS_METHODS

SESSION = HoldoutTuningSession()
if MODEL_IDS:
    HOLDOUT_MODELS = build_holdout_models_list(
        PROJECT_DIR, workflow_root=WORKFLOW_ROOT, model_ids=MODEL_IDS
    )
else:
    HOLDOUT_MODELS = build_holdout_models_list(PROJECT_DIR, workflow_root=WORKFLOW_ROOT)
holdout, PROMPT_COL, SVG_COL = prepare_holdout_and_columns(WORKFLOW_ROOT)
RANKED_PATH = PROCESSED_DIR / 'train_ranked.csv'

if POSTPROCESS_METHOD not in POSTPROCESS_METHODS:
    raise ValueError(f'Unknown POSTPROCESS_METHOD. Pick one of: {sorted(POSTPROCESS_METHODS)}')

_INFER_COMMON = dict(
    holdout_models=HOLDOUT_MODELS,
    holdout=holdout,
    prompt_col=PROMPT_COL,
    svg_col=SVG_COL,
    base_model_id=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    postprocess_method=POSTPROCESS_METHOD,
    eval_root=EVAL_ROOT,
    ranked_path=RANKED_PATH,
    use_percentile_buckets=USE_PERCENTILE_BUCKETS,
    percentile_buckets=PERCENTILE_BUCKETS,
    n_samples_per_bucket=N_SAMPLES_PER_BUCKET,
    seed=SEED,
    show_raw_metrics=SHOW_RAW_METRICS,
    outputs_dir=WORKFLOW_ROOT,
    force_regenerate=FORCE_REGENERATE_HOLDOUT_PREDICTIONS,
    project_root=PROJECT_DIR,
    workflow_root=WORKFLOW_ROOT,
    eval_kind='holdout_tuning',
)

print('Registry models (this stage filter):', len(HOLDOUT_MODELS))
for rec in HOLDOUT_MODELS:
    print(f"  [{rec['model_index']}] {rec['model_id']} | {rec['tuning_stage']}")
if len(HOLDOUT_MODELS) > MAX_MODEL_SLOTS:
    print('WARNING: len(HOLDOUT_MODELS) > MAX_MODEL_SLOTS — increase MAX_MODEL_SLOTS in the parameters cell.')"""
        ),
        md(
            """#### Generate predictions — **all models at once**

Clears the **in-memory** session only, then runs inference for every model (uses disk cache when manifest matches unless `FORCE_REGENERATE_HOLDOUT_PREDICTIONS` is True). Skip this cell if you only want per-slot generation below."""
        ),
        code(
            """SESSION.clear()
SESSION.run_all_models_inference(**_INFER_COMMON)
print('Saved under:', EVAL_ROOT)"""
        ),
        md(
            """#### Generate predictions — **one model per cell**

Run only the cells for the indices you need (1 … N). Slots beyond `len(HOLDOUT_MODELS)` print a skip message."""
        ),
    ]

    for k in range(1, max_slots + 1):
        cells.append(md(f"##### Model slot **{k}** — generation only"))
        cells.append(code(f"SESSION.run_model_inference({k}, **_INFER_COMMON)"))

    cells.append(md("#### Cross-model summary table (postprocessed metrics)"))
    cells.append(code("SESSION.refresh_summary_table()"))

    cells.append(
        md(
            """#### Display comparisons — **one section per model slot**

Run the display cell for index *k* **after** you have run generation for that index. Each section shows raw vs target (metrics, text, render) then postprocessed vs target (metrics, text, render)."""
        )
    )

    for k in range(1, max_slots + 1):
        cells.append(md(f"##### Model slot **{k}** — raw + postprocessed displays"))
        cells.append(
            code(
                f"SESSION.display_all_comparisons_for_model(\n"
                f"    {k},\n"
                f"    n_text_rows=N_TEXT_ROWS,\n"
                f"    n_render_rows=N_RENDER_ROWS,\n"
                f"    preview_chars=PREVIEW_CHARS,\n"
                f")"
            )
        )

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return nb_path


if __name__ == "__main__":
    p = generate_nb08()
    print("Wrote", p)
