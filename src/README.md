# `src` Overview

This directory contains the reusable Python modules that support the project notebooks. At a high level, the code flows from data preparation and SVG cleaning, into model-training helpers, and then into inference and submission utilities.

## Folder Map

- `core/`: Generic utilities for DataFrame preparation and runtime management.
- `svg/`: SVG-specific validation, sanitization, rendering, and feature-extraction helpers.
- `inference/`: Model-generation, postprocessing, and submission-building code.
- `eval/`: Qualitative evaluation helpers for side-by-side visual inspection.
- `training/`: Prompt-formatting, LoRA utilities, and seq2seq training helpers.

## Typical Data Flow

1. `core.dataframe` prepares prompt/SVG tables and train-validation splits.
2. `svg.cleaning` and `svg.features` sanitize raw SVG strings and compute difficulty or complexity features.
3. `training.prompts`, `training.lora.*`, and `training.seq2seq.*` format data and train models.
4. `inference.generation` and `inference.postprocess` generate SVG text and clean it for downstream use.
5. `inference.submission` validates outputs and writes the final submission CSV.
6. `eval.qualitative` and `training.lora.display` support human inspection of predictions.

## Notebook map (under `notebooks/`)

| Notebook | Role |
|----------|------|
| `01_eda_and_data_cleaning.ipynb` | EDA and cleaning |
| `02_difficulty_ranking.ipynb` | Difficulty ranking |
| `03_build_modeling_splits.ipynb` | Build train/val pool + holdout under `outputs/workflow_runs/<RUN_PROFILE_ID>/` (`FIRST_N_LABELED`, optional easy subset, `HOLDOUT_N`, reuse manifest, `workflow_layout.json`) |
| `04_lora_round1_tune.ipynb` | Round 1 tuning (`max_seq_length`, LR, `max_steps`; optional `TRAIN_FIRST_N`) |
| `05_lora_tune_rounds_2_3_4.ipynb` | Rounds 2–4 tuning (rank, dropout, batch; optional `TRAIN_FIRST_N`) |
| `06_lora_train_best_extended_tokens.ipynb` | Final `best_extended` train; **`EXTENDED_TRAIN_CONFIG`** + **`MAX_SEQ_LENGTH`** set manually from tuning |
| `07_lora_curriculum_best.ipynb` | Curriculum train; **`CURRICULUM_BASE_CONFIG`** + **`MAX_SEQ_LENGTH`** set manually from tuning |
| `08_eval_holdout_all_tuning_models.ipynb` | Holdout eval for all round1–4 registry models (optional `MODEL_IDS` filter) |
| `09_eval_holdout_post_training_models.ipynb` | Holdout eval for `best_extended` and `curriculum` registry rows in one loop |
| `10_kaggle_submission_by_model_id.ipynb` | Official test submission by registry `model_id` |
| `11_svg_postprocess_ablation.ipynb` | Postprocess ablations |
| `12_holdout_eval_leaderboard.ipynb` | Aggregate holdout metrics + tuning `eval_loss` for listed `model_id`s (CSV) |

**Recommended notebook run order:** **03** → **04** → **05** → **08** (tuning holdout eval) → **06** → **07** → **09** (post-training holdout eval) → **10**–**12**.

## `core/`

### `core/dataframe.py`
Utilities for selecting the right columns, cleaning prompt/SVG pairs, formatting seq2seq inputs, sampling subsets, splitting train and validation data, and assigning easy-versus-hard difficulty buckets.

Functions in this file are mainly used to normalize training tables and control which rows are used in experiments.

### `core/modeling_splits.py`
Builds a fixed holdout subset and train/val pool from labeled training data. Notebook **03** calls **`build_pool_and_holdout`** so cleaning, optional easy subset, and `save_split_artifacts` stay in this module. Generated notebooks pass **`WORKFLOW_ROOT`** (typically `outputs/workflow_runs/<RUN_PROFILE_ID>`) so CSVs and `split_manifest.json` live under `<WORKFLOW_ROOT>/modeling_splits/`. The legacy flat layout `outputs/modeling_splits` is replaced when using profiles; use `RUN_PROFILE_ID="default"` for a single canonical tree or copy old artifacts manually if migrating.

Helpers such as `split_artifacts_exist`, `load_existing_split_tables`, `load_split_manifest`, and `manifest_matches_params` support **reusing** saved splits when parameters match (`SEED`, `HOLDOUT_N`, `FIRST_N_LABELED`, `run_profile_id`, optional easy-subset fields, `source_csv`; notebook **03** sets `REUSE_EXISTING_SPLITS` / `FORCE_REBUILD_SPLITS`). Training notebooks **04–07** may set **`TRAIN_FIRST_N`** to use only the first *N* rows of the train split (sequential split when set; full shuffled train/val split when `None`).

### `core/workflow_layout.py`
Writes **`workflow_layout.json`** at the profile root with paths (relative to the repo) to modeling splits, models directory, registry CSV, LoRA tuning workflow dir, and evaluation subfolders. After holdout eval, **`update_workflow_layout_prediction`** records each `model_id` → postprocessed predictions CSV path under `predictions_by_model_id`.

### `core/runtime.py`
Helpers for experiment runtime behavior, including warning suppression, reproducible random seeds, and CUDA memory cleanup.

Functions in this file are used before and after training runs so notebook output stays readable and GPU memory is reclaimed between experiments.

## `svg/`

### `svg/cleaning.py`
Implements the main SVG sanitization pipeline. It checks XML validity, detects wrappers and namespaces, parses attributes, removes unsafe references and event handlers, rebuilds an allowed SVG tree, truncates oversized path-heavy SVGs, validates submission constraints, and falls back to a safe empty SVG when necessary.

Functions in this file are used anywhere the project needs submission-safe SVG text, especially after model generation.

### `svg/features.py`
Computes prompt and SVG complexity features, such as path-command counts, shape counts, grouping depth, numeric-token counts, prompt lexical counts, and weighted ranking scores.

Functions in this file are used to score examples by difficulty and to derive interpretable complexity signals for curriculum or analysis notebooks.

### `svg/rendering.py`
Renders SVG strings into images using CairoSVG and PIL, returning either PIL images or NumPy arrays with graceful failure handling.

Functions in this file are used by evaluation and display helpers that need to visualize target and predicted SVGs.

## `inference/`

### `inference/generation.py`
Causal LM generation: `generate_svg_prediction` (sanitized) and `generate_svg_raw_prediction` (raw continuation for separate postprocess).

Functions in this file are used during validation panels, notebook inference, and final submission generation. Use `generate_svg_raw_prediction` when decoded text must stay unsanitized until a separate postprocess step.

### `inference/postprocess.py`
Extracts the most likely SVG fragment from raw model output and runs it through the SVG cleaner.

Functions in this file are used immediately after generation to turn raw decoded text into valid project-compatible SVG strings.

### `inference/submission.py`
Provides the end-to-end submission pipeline: load the test CSV and adapter, generate one SVG per row, validate outputs, preview predictions, summarize throughput, and write the final CSV.

Functions in this file are used in final inference notebooks or scripts that need a submission-ready DataFrame and CSV.

## `eval/`

### `eval/qualitative.py`
Builds side-by-side matplotlib figures showing rendered target and predicted SVGs, along with a shortened prompt for context.

Functions in this file are used for quick visual sanity checks of model behavior.

### `eval/postprocess_presets.py`
Named postprocess callables (`POSTPROCESS_METHODS`) shared by holdout evaluation and submission notebooks, aligned with notebook **11**-style ablations.

### `eval/postprocess_ablation.py`
Notebook **11** helpers: `register_method`, `score_postprocess_method`, and `pick_gallery_rows` for comparing methods on a shared raw-output dataframe.

### `eval/holdout_evaluation.py`
Raw generation on the holdout CSV, optional postprocess column, enrichment for display helpers, percentile-bucket sampling, and **disk reuse** via `load_holdout_predictions_cached_or_run` plus `eval_run_manifest.json` (holdout fingerprint, `max_new_tokens`, resolved adapter path, base model id, postprocess method).

### `eval/holdout_leaderboard.py`
Builds a per-`model_id` summary table from saved `predictions_post_*.csv` files under the holdout eval directories (notebooks **08** and **09**), registry training config fields, and `eval_loss` from `round1_results.csv`–`round4_results.csv` when `registry_model_id` is present. Pass **`workflow_root`** so eval roots, tuning CSV directory, and registry resolve under the same profile as notebook **03**. Used by notebook **12**.

### `eval/holdout_tuning_notebook.py`
Session helpers for notebook **08**: build the registry model list (`workflow_root` scopes `model_registry.csv`), run/cache inference per model index, batch or per-slot generation, and grouped display (raw + postprocessed) like legacy notebook **09**. When `project_root` and `workflow_root` are passed to `run_model_inference`, prediction paths are merged into `workflow_layout.json`.

## `training/`

### `training/prompts.py`
Formats prompt/SVG examples into the chat-style instruction strings used for LoRA supervised fine-tuning and builds DataFrames containing those formatted examples.

Functions in this file are used before LoRA training and sometimes to create inference-style prompts.

### `training/lora/display.py`
Notebook-oriented display helpers for summarizing prediction panels, comparing multiple models, showing raw SVG text side by side, and rendering target-versus-prediction images.

Functions in this file are used after generation to make experiment comparison easier inside notebooks.

### `training/lora/eval.py`
Evaluates generated SVGs by checking XML validity, render success, path counts, and submission validity, then summarizes those metrics over a sampled validation panel.

Functions in this file are used during and after training to measure qualitative generation quality beyond loss alone.

### `training/lora/experiments.py`
Contains the main LoRA experiment orchestration code. It builds instruction datasets, configures SFT training arguments, logs device placement, saves adapters, runs baseline and metric-aware experiments, and manages curriculum training stages.

Functions and classes in this file are used to launch, monitor, and export LoRA training runs from notebooks. New helpers include `run_single_experiment_eval_loss_early_stop` (eval-loss early stopping only, registry export) and `run_curriculum_experiment_eval_loss_only`.

### `training/lora/registry.py`
Copies adapters to `<models_root>/lora_model_id_<id>/` and appends `model_registry.csv` under the same root (default `outputs/models`; workflow notebooks pass `WORKFLOW_ROOT / "models"`). Stored `adapter_dir` paths remain relative to the repository root.

### `training/lora/tuning_utils.py`
Picks tuning winners by lowest `eval_loss` and appends per-run CSV rows.

### `training/lora/modeling.py`
Loads base causal language models and tokenizers, creates quantization configs, applies LoRA adapters, and reloads saved adapters for inference.

Functions in this file are used wherever a LoRA-capable model must be created or restored.

### `training/seq2seq/dataset.py`
Defines the `SVGSeq2SeqDataset` class, which tokenizes prompt/SVG pairs and prepares masked labels for supervised seq2seq training.

The class in this file is used to feed tokenized examples into PyTorch dataloaders.

### `training/seq2seq/modeling.py`
Loads a pretrained T5 model and tokenizer for seq2seq experiments.

Functions in this file are used to bootstrap smaller encoder-decoder baselines.

### `training/seq2seq/preprocess.py`
Re-exports the core seq2seq preprocessing helpers from `core.dataframe`.

This file is used to keep notebook imports simple and grouped under the `training.seq2seq` namespace.

### `training/seq2seq/train_loop.py`
Implements a lightweight PyTorch training loop for seq2seq models, including one-epoch training, one-epoch evaluation, progress bars, optional checkpoint saving, and per-epoch history tracking.

Functions in this file are used for manual seq2seq experiments outside the LoRA training stack.
