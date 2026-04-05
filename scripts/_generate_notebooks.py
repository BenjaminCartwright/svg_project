"""One-off generator for LoRA workflow notebooks (03–13). Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def save(nb, name: str):
    path = NB / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Wrote", path)


def md(s: str):
    return nbf.v4.new_markdown_cell(s)


def code(s: str):
    return nbf.v4.new_code_cell(s)


# --- 03 build modeling splits only ---
nb03_build = nbf.v4.new_notebook()
nb03_build["cells"] = [
    md(
        """# 03 — Build modeling splits (train/val pool + holdout)

**Summary:** Load labeled training data (ranked, clean, or raw), clean prompts/SVGs, optionally merge **difficulty metadata** from `train_ranked.csv` when needed, optionally keep only the **easiest `EASY_SUBSET_FRAC`** of rows (`USE_EASY_SUBSET`), then optionally the **first `FIRST_N_LABELED` rows**, then reserve **`HOLDOUT_N` rows** for holdout. Saves under `outputs/workflow_runs/<RUN_PROFILE_ID>/modeling_splits/` plus `workflow_layout.json` at the profile root.

**Prerequisites:** Notebooks **01** and **02** if you use ranked data or easy subset without difficulty columns in the chosen file; otherwise raw `train.csv` is enough.

**Profiles:** Set **`RUN_PROFILE_ID`** (filesystem-safe slug) when you change `SEED`, `HOLDOUT_N`, `FIRST_N_LABELED`, or easy-subset settings so artifacts do not collide. Use `RUN_PROFILE_ID="default"` for a single canonical tree.

**Stable holdout:** With `REUSE_EXISTING_SPLITS = True` (default), reruns load existing CSVs when the manifest matches parameters. Set `FORCE_REBUILD_SPLITS = True` to overwrite.

**Next:** Run notebook **04** using the same `RUN_PROFILE_ID` / `WORKFLOW_ROOT` as here."""
    ),
    md("#### Colab: mount Google Drive (optional)"),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    md("#### Install dependencies"),
    code(
        "!pip -q install transformers peft accelerate bitsandbytes datasets trl cairosvg pillow scikit-learn lxml pandas"
    ),
    md("#### Imports, paths"),
    code(
        """import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
# For local runs, uncomment and set:
# PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RAW_DIR = PROJECT_DIR / 'data' / 'raw'
INTERIM_DIR = PROJECT_DIR / 'data' / 'interim'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

print('PROJECT_DIR:', PROJECT_DIR)"""
    ),
    md("#### Parameters"),
    code(
        """SEED = 42
HOLDOUT_N = 100
# After cleaning: use only the first N rows in table order (None = all rows).
FIRST_N_LABELED = None  # e.g. 2000

# Isolate all downstream artifacts (splits, models, tuning CSVs, evals) under this profile.
RUN_PROFILE_ID = 'default'  # change when SEED / HOLDOUT_N / FIRST_N / easy subset / data source changes
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID

# Easiest fraction of difficulty-ranked rows (after optional merge from train_ranked.csv).
USE_EASY_SUBSET = False
EASY_SUBSET_FRAC = 0.2

REUSE_EXISTING_SPLITS = True
FORCE_REBUILD_SPLITS = False

print(
    'WORKFLOW_ROOT:', WORKFLOW_ROOT,
    '| HOLDOUT_N:', HOLDOUT_N,
    '| FIRST_N_LABELED:', FIRST_N_LABELED,
    '| USE_EASY_SUBSET:', USE_EASY_SUBSET,
    '| REUSE_EXISTING_SPLITS:', REUSE_EXISTING_SPLITS,
    '| FORCE_REBUILD_SPLITS:', FORCE_REBUILD_SPLITS,
)"""
    ),
    md("#### Load data, clean, build holdout + pool, save splits"),
    code(
        """from src.core.modeling_splits import (
    build_pool_and_holdout,
    load_existing_split_tables,
    load_split_manifest,
    manifest_matches_params,
    split_artifacts_exist,
)
from src.core.workflow_layout import write_workflow_layout_stub

RANKED_PATH = PROCESSED_DIR / 'train_ranked.csv'
CLEAN_PATH = INTERIM_DIR / 'train_clean_basic.csv'
RAW_TRAIN_PATH = RAW_DIR / 'train.csv'

if RANKED_PATH.exists():
    data_path = RANKED_PATH
elif CLEAN_PATH.exists():
    data_path = CLEAN_PATH
else:
    data_path = RAW_TRAIN_PATH


if FORCE_REBUILD_SPLITS or not REUSE_EXISTING_SPLITS or not split_artifacts_exist(WORKFLOW_ROOT):
    train_val_pool, holdout_df = build_pool_and_holdout(
        data_path=data_path,
        workflow_root=WORKFLOW_ROOT,
        seed=SEED,
        holdout_n=HOLDOUT_N,
        run_profile_id=str(RUN_PROFILE_ID),
        use_easy_subset=bool(USE_EASY_SUBSET),
        easy_subset_frac=float(EASY_SUBSET_FRAC),
        first_n_labeled=FIRST_N_LABELED,
        ranked_path=RANKED_PATH,
    )
else:
    manifest = load_split_manifest(WORKFLOW_ROOT)
    ok, msg = manifest_matches_params(
        manifest,
        seed=SEED,
        holdout_n=HOLDOUT_N,
        source_csv=str(data_path),
        first_n_labeled=FIRST_N_LABELED,
        run_profile_id=str(RUN_PROFILE_ID),
        use_easy_subset=bool(USE_EASY_SUBSET),
        easy_subset_frac=float(EASY_SUBSET_FRAC) if USE_EASY_SUBSET else None,
    )
    if not ok:
        raise ValueError(
            f'Cannot reuse splits: {msg}. Set FORCE_REBUILD_SPLITS = True to regenerate, '
            'or align SEED / HOLDOUT_N / FIRST_N_LABELED / RUN_PROFILE_ID / easy subset / data source.'
        )
    train_val_pool, holdout_df = load_existing_split_tables(WORKFLOW_ROOT)
    print('Reused splits from disk (holdout unchanged).', msg)

_man = load_split_manifest(WORKFLOW_ROOT)
write_workflow_layout_stub(
    PROJECT_DIR,
    WORKFLOW_ROOT,
    run_profile_id=str(RUN_PROFILE_ID),
    split_manifest=_man,
)

print('Pool rows:', len(train_val_pool), '| Holdout:', len(holdout_df))"""
    ),
]
save(nb03_build, "03_build_modeling_splits.ipynb")

# --- 04 round 1 only ---
nb04 = nbf.v4.new_notebook()
nb04["cells"] = [
    md(
        """# 04 — Round 1 tuning (length, LR, max_steps)

**Summary:** Loads the train/val pool from notebook **03** (same **`WORKFLOW_ROOT`** / `RUN_PROFILE_ID`). Sweeps `max_seq_length`, `learning_rate`, and `max_steps`; early stopping on **`eval_loss` only**; each run registers under `<WORKFLOW_ROOT>/models/lora_model_id_<id>/` and appends `<WORKFLOW_ROOT>/lora_tuning_workflow/round1_results.csv`.

**Prerequisites:** Notebook **03** with matching `RUN_PROFILE_ID`. Notebooks **01** and **02** if you use ranked source data.

**Training subset:** `TRAIN_FIRST_N` — if set, uses only the **first N rows of the train split** (after pool load); train/val split uses `shuffle=False` so order matches the pool CSV; validation is then subsampled to about **`VAL_FRAC * N`** rows from the held-out val split. If `None`, the full train split and full val split from `VAL_FRAC` on the pool are used with shuffled train/val (default behavior).

**Parameters:** `RUN_PROFILE_ID`, `WORKFLOW_ROOT`, `VAL_FRAC`, `TRAIN_FIRST_N`, grids, `ROUND1_BASE_CONFIG`.

**Recommended run order:** 03 → 04 → 05 → **08** (holdout eval for round1–4 models) → 06 → 07 → **09** (holdout eval for `best_extended` / `curriculum`) → 10–12."""
    ),
    md("#### Colab: mount Google Drive (optional)"),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    md("#### Install dependencies"),
    code(
        "!pip -q install transformers peft accelerate bitsandbytes datasets trl cairosvg pillow scikit-learn lxml pandas"
    ),
    md("#### Imports, paths"),
    code(
        """import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
EXPERIMENT_ROOT = WORKFLOW_ROOT / 'lora_tuning_workflow'
ROUND1_RESULTS_PATH = EXPERIMENT_ROOT / 'round1_results.csv'

print('PROJECT_DIR:', PROJECT_DIR)
print('WORKFLOW_ROOT:', WORKFLOW_ROOT)
print('cuda:', torch.cuda.is_available())"""
    ),
    md("#### Parameters"),
    code(
        """SEED = 42
VAL_FRAC = 0.10
# First N rows of the train split only (None = full train). When set, train/val split is sequential (shuffle=False).
TRAIN_FIRST_N = None  # e.g. 400

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

MAX_SEQ_LENGTH_GRID = [1024, 1536]
LEARNING_RATE_GRID = [1e-4, 2e-4]
MAX_STEPS_GRID = [120, 200]

ROUND1_BASE_CONFIG = {
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.0,
    'eval_steps': 40,
}

EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
print('VAL_FRAC:', VAL_FRAC, '| TRAIN_FIRST_N:', TRAIN_FIRST_N)"""
    ),
    md("#### Load pool, train/val split"),
    code(
        """from src.core.dataframe import choose_first_existing, train_val_split_df
from src.core.modeling_splits import load_train_val_pool

train_val_pool = load_train_val_pool(WORKFLOW_ROOT)
PROMPT_COL = choose_first_existing(train_val_pool, ['prompt', 'description', 'text'], 'pool')
SVG_COL = choose_first_existing(train_val_pool, ['svg', 'svg_code', 'target', 'label'], 'pool')
_shuffle = TRAIN_FIRST_N is None
train_df, val_df = train_val_split_df(
    train_val_pool, val_frac=VAL_FRAC, seed=SEED, shuffle=_shuffle
)
if TRAIN_FIRST_N is not None:
    train_df = train_df.iloc[: int(TRAIN_FIRST_N)].reset_index(drop=True)
    n_val = min(len(val_df), max(1, int(VAL_FRAC * TRAIN_FIRST_N)))
    val_df = val_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)
print('Train:', len(train_df), 'Val:', len(val_df))"""
    ),
    md("#### Round 1 grid (register each run)"),
    code(
        """from src.core.runtime import cleanup_memory, set_seed
from src.training.lora.experiments import run_single_experiment_eval_loss_early_stop
from src.training.lora.tuning_utils import append_round_results_csv

set_seed(SEED)
cleanup_memory()

if ROUND1_RESULTS_PATH.exists():
    ROUND1_RESULTS_PATH.unlink()

run_idx = 0
for max_seq_length in MAX_SEQ_LENGTH_GRID:
    for lr in LEARNING_RATE_GRID:
        for max_steps in MAX_STEPS_GRID:
            run_idx += 1
            cfg = {**ROUND1_BASE_CONFIG, 'learning_rate': lr, 'max_steps': int(max_steps)}
            run_name = f'r1_{run_idx:03d}_msl{max_seq_length}_lr{lr}_steps{max_steps}_seed{SEED}'
            print('===', run_name)
            summary, _, run_dir, reg_id = run_single_experiment_eval_loss_early_stop(
                run_name=run_name,
                config=cfg,
                train_df=train_df,
                val_df=val_df,
                prompt_col=PROMPT_COL,
                svg_col=SVG_COL,
                model_id=MODEL_ID,
                max_seq_length=max_seq_length,
                root_dir=EXPERIMENT_ROOT,
                project_root=PROJECT_DIR,
                tuning_stage='round1',
                curriculum=False,
                seed=SEED,
                eval_steps=cfg.get('eval_steps'),
                notes='round1 grid',
                models_root=MODELS_ROOT,
            )
            row = {**summary, 'max_seq_length': max_seq_length, 'tuning_stage': 'round1'}
            append_round_results_csv(ROUND1_RESULTS_PATH, row)
            cleanup_memory()

r1 = pd.read_csv(ROUND1_RESULTS_PATH)
print('Round 1 done. Rows:', len(r1))
# Inspection only — does not select models for later rounds (see notebook 05).
display(r1.sort_values('eval_loss').head(10))"""
    ),
]
save(nb04, "04_lora_round1_tune.ipynb")

# --- 05 rounds 2–4 ---
nb05 = nbf.v4.new_notebook()
nb05["cells"] = [
    md(
        """# 05 — Rounds 2, 3, and 4 hyperparameter tuning

**Summary:** Three sequential sweeps in one notebook. Each round reads the **previous round’s results CSV** and fixes hyperparameters for the next sweep:

- **Default (`USE_AUTO_WINNER_FROM_PRIOR_ROUND = False`):** You set **`MANUAL_BASE_AFTER_ROUND1`**, **`MANUAL_BASE_AFTER_ROUND2`**, and **`MANUAL_BASE_AFTER_ROUND3`** (copy field-for-field from the `round1` / `round2` / `round3` CSV row you want to continue from) plus **`MANUAL_MAX_SEQ_LENGTH`** for all rounds.
- **Optional auto chain (`USE_AUTO_WINNER_FROM_PRIOR_ROUND = True`):** After each round, the next base config is **`pick_winner_by_eval_loss`** on that round’s results (lowest `eval_loss`).

**Sweeps:** (1) **Round 2:** `lora_r` and `lora_alpha = 2 * lora_r` — (2) **Round 3:** `lora_dropout` — (3) **Round 4:** batch / accumulation (optional: `TARGET_EFFECTIVE_BATCH`).

**Prerequisites:** Notebooks **03** (splits) and **04** (`round1_results.csv`).

**Recommended run order:** 03 → 04 → 05 → **08** (holdout eval for round1–4) → 06 → 07 → **09** (post-train holdout eval) → 10–12.

**Outputs:** `round2_results.csv`, `round3_results.csv`, `round4_results.csv` under `<WORKFLOW_ROOT>/lora_tuning_workflow/` (set `RUN_PROFILE_ID` to match notebook **03**)."""
    ),
    md("#### Colab: mount (optional)"),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    md("#### Install"),
    code("!pip -q install transformers peft accelerate bitsandbytes datasets trl cairosvg pillow scikit-learn lxml pandas"),
    md("#### Imports"),
    code(
        """import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RAW_DIR = PROJECT_DIR / 'data' / 'raw'
INTERIM_DIR = PROJECT_DIR / 'data' / 'interim'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
EXPERIMENT_ROOT = WORKFLOW_ROOT / 'lora_tuning_workflow'
ROUND1_PATH = EXPERIMENT_ROOT / 'round1_results.csv'
ROUND2_PATH = EXPERIMENT_ROOT / 'round2_results.csv'
ROUND3_PATH = EXPERIMENT_ROOT / 'round3_results.csv'
ROUND4_PATH = EXPERIMENT_ROOT / 'round4_results.csv'

print('WORKFLOW_ROOT:', WORKFLOW_ROOT)
print('cuda:', torch.cuda.is_available())"""
    ),
    md("#### Parameters"),
    code(
        """SEED = 42
VAL_FRAC = 0.10
TRAIN_FIRST_N = None  # first N rows of train split; None = full train (shuffled split)

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

# False (default): use MANUAL_* dicts below for bases between rounds. True: pick_winner_by_eval_loss after each round.
USE_AUTO_WINNER_FROM_PRIOR_ROUND = False
# Used when USE_AUTO_WINNER_FROM_PRIOR_ROUND is False (fixed across rounds 2–4).
MANUAL_MAX_SEQ_LENGTH = 1536
# Paste from round1_results.csv / round2_results.csv / round3_results.csv rows you want as the starting point for the next round.
MANUAL_BASE_AFTER_ROUND1 = {
    'learning_rate': 2e-4,
    'max_steps': 200,
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.0,
    'eval_steps': 40,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
}
MANUAL_BASE_AFTER_ROUND2 = {
    'learning_rate': 2e-4,
    'max_steps': 200,
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.0,
    'eval_steps': 40,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
}
MANUAL_BASE_AFTER_ROUND3 = {
    'learning_rate': 2e-4,
    'max_steps': 200,
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.0,
    'eval_steps': 40,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
}

ROUND2_LORA_R_GRID = [8, 16, 32]
ROUND3_DROPOUT_GRID = [0.0, 0.05, 0.10]
# (per_device_batch, grad_accum) pairs; set TARGET_EFFECTIVE_BATCH to None to disable check
ROUND4_BATCH_GRID = [(1, 8), (2, 4), (4, 2)]
TARGET_EFFECTIVE_BATCH = 8  # set to None to try every pair in ROUND4_BATCH_GRID"""
    ),
    md("#### Load pool + previous results, rebuild train/val"),
    code(
        """from src.core.dataframe import choose_first_existing, train_val_split_df
from src.core.modeling_splits import load_train_val_pool
from src.training.lora.tuning_utils import pick_winner_by_eval_loss

train_val_pool = load_train_val_pool(WORKFLOW_ROOT)
PROMPT_COL = choose_first_existing(train_val_pool, ['prompt', 'description', 'text'], 'pool')
SVG_COL = choose_first_existing(train_val_pool, ['svg', 'svg_code', 'target', 'label'], 'pool')
_shuffle = TRAIN_FIRST_N is None
train_df, val_df = train_val_split_df(
    train_val_pool, val_frac=VAL_FRAC, seed=SEED, shuffle=_shuffle
)
if TRAIN_FIRST_N is not None:
    train_df = train_df.iloc[: int(TRAIN_FIRST_N)].reset_index(drop=True)
    n_val = min(len(val_df), max(1, int(VAL_FRAC * TRAIN_FIRST_N)))
    val_df = val_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)

r1 = pd.read_csv(ROUND1_PATH)
if USE_AUTO_WINNER_FROM_PRIOR_ROUND:
    w1 = pick_winner_by_eval_loss(r1)
    print('Round 1 winner eval_loss:', w1['eval_loss'])
    base = {
        'learning_rate': float(w1['learning_rate']),
        'max_steps': int(w1['max_steps']),
        'warmup_ratio': float(w1.get('warmup_ratio', 0.05)),
        'lr_scheduler_type': str(w1.get('lr_scheduler_type', 'cosine')),
        'early_stopping_patience': int(w1.get('early_stopping_patience', 3)),
        'early_stopping_min_delta': float(w1.get('early_stopping_min_delta', 0.0)),
        'eval_steps': int(w1.get('eval_steps', 40)),
        'lora_r': int(w1['lora_r']),
        'lora_alpha': int(w1['lora_alpha']),
        'lora_dropout': float(w1['lora_dropout']),
        'per_device_train_batch_size': int(w1['per_device_train_batch_size']),
        'gradient_accumulation_steps': int(w1['gradient_accumulation_steps']),
    }
    max_seq_length = int(w1['max_seq_length'])
else:
    base = {k: MANUAL_BASE_AFTER_ROUND1[k] for k in MANUAL_BASE_AFTER_ROUND1}
    max_seq_length = int(MANUAL_MAX_SEQ_LENGTH)
    print('Manual base for round 2 (from MANUAL_BASE_AFTER_ROUND1):', max_seq_length, base)
print('Fixed max_seq_length:', max_seq_length, base)"""
    ),
    md("#### Round 2 — sweep LoRA rank"),
    code(
        """from src.core.runtime import cleanup_memory, set_seed
from src.training.lora.experiments import run_single_experiment_eval_loss_early_stop
from src.training.lora.tuning_utils import append_round_results_csv

set_seed(SEED)
cleanup_memory()
if ROUND2_PATH.exists():
    ROUND2_PATH.unlink()

for i, r in enumerate(ROUND2_LORA_R_GRID, start=1):
    cfg = {**base, 'lora_r': int(r), 'lora_alpha': int(2 * r)}
    run_name = f'r2_{i:03d}_r{r}_seed{SEED}'
    summary, _, run_dir, reg_id = run_single_experiment_eval_loss_early_stop(
        run_name=run_name,
        config=cfg,
        train_df=train_df,
        val_df=val_df,
        prompt_col=PROMPT_COL,
        svg_col=SVG_COL,
        model_id=MODEL_ID,
        max_seq_length=max_seq_length,
        root_dir=EXPERIMENT_ROOT,
        project_root=PROJECT_DIR,
        tuning_stage='round2',
        curriculum=False,
        seed=SEED,
        eval_steps=cfg.get('eval_steps'),
        notes='round2 lora_r',
        models_root=MODELS_ROOT,
    )
    row = {**summary, 'max_seq_length': max_seq_length, 'tuning_stage': 'round2'}
    append_round_results_csv(ROUND2_PATH, row)
    cleanup_memory()

r2 = pd.read_csv(ROUND2_PATH)
if USE_AUTO_WINNER_FROM_PRIOR_ROUND:
    w2 = pick_winner_by_eval_loss(r2)
    base2 = {
        'learning_rate': float(w2['learning_rate']),
        'max_steps': int(w2['max_steps']),
        'warmup_ratio': float(w2.get('warmup_ratio', 0.05)),
        'lr_scheduler_type': str(w2.get('lr_scheduler_type', 'cosine')),
        'early_stopping_patience': int(w2.get('early_stopping_patience', 3)),
        'early_stopping_min_delta': float(w2.get('early_stopping_min_delta', 0.0)),
        'eval_steps': int(w2.get('eval_steps', 40)),
        'lora_r': int(w2['lora_r']),
        'lora_alpha': int(w2['lora_alpha']),
        'lora_dropout': float(w2['lora_dropout']),
        'per_device_train_batch_size': int(w2['per_device_train_batch_size']),
        'gradient_accumulation_steps': int(w2['gradient_accumulation_steps']),
    }
    print('Round 2 winner:', base2)
else:
    base2 = {k: MANUAL_BASE_AFTER_ROUND2[k] for k in MANUAL_BASE_AFTER_ROUND2}
    print('Manual base for round 3 (MANUAL_BASE_AFTER_ROUND2):', base2)"""
    ),
    md("#### Round 3 — sweep dropout"),
    code(
        """set_seed(SEED)
cleanup_memory()
if ROUND3_PATH.exists():
    ROUND3_PATH.unlink()

for i, do in enumerate(ROUND3_DROPOUT_GRID, start=1):
    cfg = {**base2, 'lora_dropout': float(do)}
    run_name = f'r3_{i:03d}_do{do}_seed{SEED}'
    summary, _, run_dir, reg_id = run_single_experiment_eval_loss_early_stop(
        run_name=run_name,
        config=cfg,
        train_df=train_df,
        val_df=val_df,
        prompt_col=PROMPT_COL,
        svg_col=SVG_COL,
        model_id=MODEL_ID,
        max_seq_length=max_seq_length,
        root_dir=EXPERIMENT_ROOT,
        project_root=PROJECT_DIR,
        tuning_stage='round3',
        curriculum=False,
        seed=SEED,
        eval_steps=cfg.get('eval_steps'),
        notes='round3 dropout',
        models_root=MODELS_ROOT,
    )
    row = {**summary, 'max_seq_length': max_seq_length, 'tuning_stage': 'round3'}
    append_round_results_csv(ROUND3_PATH, row)
    cleanup_memory()

r3 = pd.read_csv(ROUND3_PATH)
if USE_AUTO_WINNER_FROM_PRIOR_ROUND:
    w3 = pick_winner_by_eval_loss(r3)
    base3 = {
        'learning_rate': float(w3['learning_rate']),
        'max_steps': int(w3['max_steps']),
        'warmup_ratio': float(w3.get('warmup_ratio', 0.05)),
        'lr_scheduler_type': str(w3.get('lr_scheduler_type', 'cosine')),
        'early_stopping_patience': int(w3.get('early_stopping_patience', 3)),
        'early_stopping_min_delta': float(w3.get('early_stopping_min_delta', 0.0)),
        'eval_steps': int(w3.get('eval_steps', 40)),
        'lora_r': int(w3['lora_r']),
        'lora_alpha': int(w3['lora_alpha']),
        'lora_dropout': float(w3['lora_dropout']),
        'per_device_train_batch_size': int(w3['per_device_train_batch_size']),
        'gradient_accumulation_steps': int(w3['gradient_accumulation_steps']),
    }
    print('Round 3 winner:', base3)
else:
    base3 = {k: MANUAL_BASE_AFTER_ROUND3[k] for k in MANUAL_BASE_AFTER_ROUND3}
    print('Manual base for round 4 (MANUAL_BASE_AFTER_ROUND3):', base3)"""
    ),
    md("#### Round 4 — sweep batch / accumulation"),
    code(
        """set_seed(SEED)
cleanup_memory()
if ROUND4_PATH.exists():
    ROUND4_PATH.unlink()

j = 0
for pbs, gas in ROUND4_BATCH_GRID:
    if TARGET_EFFECTIVE_BATCH is not None and pbs * gas != TARGET_EFFECTIVE_BATCH:
        continue
    j += 1
    cfg = {**base3, 'per_device_train_batch_size': int(pbs), 'gradient_accumulation_steps': int(gas)}
    run_name = f'r4_{j:03d}_bs{pbs}_ga{gas}_seed{SEED}'
    summary, _, run_dir, reg_id = run_single_experiment_eval_loss_early_stop(
        run_name=run_name,
        config=cfg,
        train_df=train_df,
        val_df=val_df,
        prompt_col=PROMPT_COL,
        svg_col=SVG_COL,
        model_id=MODEL_ID,
        max_seq_length=max_seq_length,
        root_dir=EXPERIMENT_ROOT,
        project_root=PROJECT_DIR,
        tuning_stage='round4',
        curriculum=False,
        seed=SEED,
        eval_steps=cfg.get('eval_steps'),
        notes='round4 batch',
        models_root=MODELS_ROOT,
    )
    row = {**summary, 'max_seq_length': max_seq_length, 'tuning_stage': 'round4'}
    append_round_results_csv(ROUND4_PATH, row)
    cleanup_memory()

r4 = pd.read_csv(ROUND4_PATH)
print(r4.sort_values('eval_loss').head())
print('Done rounds 2–4.')"""
    ),
]
save(nb05, "05_lora_tune_rounds_2_3_4.ipynb")

# --- 06 best extended ---
nb06 = nbf.v4.new_notebook()
nb06["cells"] = [
    md(
        """# 06 — Train best config (manual hyperparameters, registry)

**Summary:** One final LoRA fine-tune with **hyperparameters you set** in this notebook (copy from your tuning results in **04–05**). Registers under `tuning_stage='best_extended'`.

**Prerequisites:** Notebook **03** (splits); complete hypertuning elsewhere, then paste winning values into **`EXTENDED_TRAIN_CONFIG`** and **`MAX_SEQ_LENGTH`** below.

**Recommended run order:** 03 → 04 → 05 → **08** (holdout eval for tuning models) → **06** (this notebook) → 07 → **09** (post-train holdout eval) → 10–12."""
    ),
    md("#### Colab / install"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n!pip -q install transformers peft accelerate bitsandbytes datasets trl cairosvg pillow scikit-learn lxml pandas"),
    md("#### Imports + parameters"),
    code(
        """import sys
from pathlib import Path

import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
EXPERIMENT_ROOT = WORKFLOW_ROOT / 'lora_tuning_workflow'

SEED = 42
VAL_FRAC = 0.10
TRAIN_FIRST_N = None  # first N rows of train split; None = full train (shuffled split)

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

# Paste from your best tuning row (e.g. round4_results.csv), then adjust.
MAX_SEQ_LENGTH = 1536

EXTENDED_TRAIN_CONFIG = {
    'learning_rate': 2e-4,
    'max_steps': 200,
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'early_stopping_patience': 3,
    'early_stopping_min_delta': 0.0,
    'eval_steps': 40,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
}

print('WORKFLOW_ROOT:', WORKFLOW_ROOT)
print('cuda:', torch.cuda.is_available())"""
    ),
    md("#### Load data, train final extended run"),
    code(
        """from src.core.dataframe import choose_first_existing, train_val_split_df
from src.core.modeling_splits import load_train_val_pool
from src.core.runtime import cleanup_memory, set_seed
from src.training.lora.experiments import run_single_experiment_eval_loss_early_stop

set_seed(SEED)
cleanup_memory()

train_val_pool = load_train_val_pool(WORKFLOW_ROOT)
PROMPT_COL = choose_first_existing(train_val_pool, ['prompt', 'description', 'text'], 'pool')
SVG_COL = choose_first_existing(train_val_pool, ['svg', 'svg_code', 'target', 'label'], 'pool')
_shuffle = TRAIN_FIRST_N is None
train_df, val_df = train_val_split_df(
    train_val_pool, val_frac=VAL_FRAC, seed=SEED, shuffle=_shuffle
)
if TRAIN_FIRST_N is not None:
    train_df = train_df.iloc[: int(TRAIN_FIRST_N)].reset_index(drop=True)
    n_val = min(len(val_df), max(1, int(VAL_FRAC * TRAIN_FIRST_N)))
    val_df = val_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)

cfg = dict(EXTENDED_TRAIN_CONFIG)
max_seq_length = int(MAX_SEQ_LENGTH)

run_name = f'best_extended_msl{max_seq_length}_seed{SEED}'
summary, _, run_dir, reg_id = run_single_experiment_eval_loss_early_stop(
    run_name=run_name,
    config=cfg,
    train_df=train_df,
    val_df=val_df,
    prompt_col=PROMPT_COL,
    svg_col=SVG_COL,
    model_id=MODEL_ID,
    max_seq_length=max_seq_length,
    root_dir=EXPERIMENT_ROOT,
    project_root=PROJECT_DIR,
    tuning_stage='best_extended',
    curriculum=False,
    seed=SEED,
    eval_steps=cfg.get('eval_steps'),
    notes='best extended (manual hparams)',
    models_root=MODELS_ROOT,
)

print('registry_model_id:', reg_id)
print('summary:', summary)"""
    ),
]
save(nb06, "06_lora_train_best_extended_tokens.ipynb")

# --- 07 curriculum ---
nb07 = nbf.v4.new_notebook()
nb07["cells"] = [
    md(
        """# 07 — Curriculum training (manual hyperparameters)

**Summary:** Multi-stage curriculum over the **difficulty-sorted** train/val pool. **Early stopping on `eval_loss` only** per stage. LoRA/optimizer settings come from **`CURRICULUM_BASE_CONFIG`** (you paste from tuning results). Registers one adapter with `curriculum=True`.

**Prerequisites:** Notebook **03** (splits); notebook **02** (or ranked columns) helps difficulty sorting. Set **`CURRICULUM_BASE_CONFIG`** and **`MAX_SEQ_LENGTH`** below after hypertuning.

**Recommended run order:** 03 → 04 → 05 → **08** → 06 → **07** (this notebook) → **09** → 10–12.

**Training subset:** `TRAIN_FIRST_N` caps rows on the **sorted train split** (first N in difficulty order within that split) when set; uses sequential train/val split from the pool when set; validation is then subsampled to about **`VAL_FRAC * N`** rows from the held-out val split."""
    ),
    md("#### Colab / install"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n!pip -q install transformers peft accelerate bitsandbytes datasets trl cairosvg pillow scikit-learn lxml pandas"),
    md("#### Imports + parameters"),
    code(
        """import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
EXPERIMENT_ROOT = WORKFLOW_ROOT / 'lora_tuning_workflow'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'

SEED = 42
VAL_FRAC = 0.10
TRAIN_FIRST_N = None  # first N rows of sorted train split; None = full train (shuffled split)

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
MAX_SEQ_LENGTH = 1536

CURRICULUM_STAGE_LABELS = ['stage1_easy', 'stage2_easy_med', 'stage3_all']
CURRICULUM_STAGE_FRACS = [0.25, 0.55, 1.0]
CURRICULUM_STAGE_MAX_STEPS = [80, 80, 120]
EVAL_STEPS = 40
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0

# Per-stage step counts override config['max_steps'] during training; keep max_steps >= max(CURRICULUM_STAGE_MAX_STEPS) if you change stages.
CURRICULUM_BASE_CONFIG = {
    'learning_rate': 2e-4,
    'max_steps': max(CURRICULUM_STAGE_MAX_STEPS),
    'warmup_ratio': 0.05,
    'lr_scheduler_type': 'cosine',
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
}

print('WORKFLOW_ROOT:', WORKFLOW_ROOT)
print('cuda:', torch.cuda.is_available())"""
    ),
    md("#### Build sorted pool + run curriculum"),
    code(
        """from src.core.dataframe import choose_first_existing, sort_by_difficulty, train_val_split_df
from src.core.modeling_splits import load_train_val_pool
from src.core.runtime import cleanup_memory, set_seed
from src.training.lora.experiments import run_curriculum_experiment_eval_loss_only

set_seed(SEED)
cleanup_memory()

pool = load_train_val_pool(WORKFLOW_ROOT)
ranked_path = PROCESSED_DIR / 'train_ranked.csv'
if ranked_path.exists():
    ranked = pd.read_csv(ranked_path)
    id_set = set(pool['id'].astype(str))
    extra = ranked[ranked['id'].astype(str).isin(id_set)]
    pool = pool.merge(extra[['id'] + [c for c in ranked.columns if c not in pool.columns and c != 'id']], on='id', how='left')
pool = sort_by_difficulty(pool)

PROMPT_COL = choose_first_existing(pool, ['prompt', 'description', 'text'], 'pool')
SVG_COL = choose_first_existing(pool, ['svg', 'svg_code', 'target', 'label'], 'pool')
_shuffle = TRAIN_FIRST_N is None
train_df, val_df = train_val_split_df(
    pool, val_frac=VAL_FRAC, seed=SEED, shuffle=_shuffle
)
train_df = sort_by_difficulty(train_df)
if TRAIN_FIRST_N is not None:
    train_df = train_df.iloc[: int(TRAIN_FIRST_N)].reset_index(drop=True)
    n_val = min(len(val_df), max(1, int(VAL_FRAC * TRAIN_FIRST_N)))
    val_df = val_df.sample(n=n_val, random_state=SEED).reset_index(drop=True)

config = dict(CURRICULUM_BASE_CONFIG)
max_sl = int(MAX_SEQ_LENGTH)

run_name = f'curriculum_seed{SEED}'
summary, gen_df, run_dir, stage_df, reg_id = run_curriculum_experiment_eval_loss_only(
    run_name=run_name,
    config=config,
    reduced_train_df=train_df,
    val_df=val_df,
    prompt_col=PROMPT_COL,
    svg_col=SVG_COL,
    model_id=MODEL_ID,
    max_seq_length=max_sl,
    root_dir=EXPERIMENT_ROOT,
    project_root=PROJECT_DIR,
    stage_fracs=CURRICULUM_STAGE_FRACS,
    stage_labels=CURRICULUM_STAGE_LABELS,
    stage_max_steps=CURRICULUM_STAGE_MAX_STEPS,
    eval_steps=EVAL_STEPS,
    seed=SEED,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    notes='curriculum (manual hparams)',
    models_root=MODELS_ROOT,
)
print('registry_model_id:', reg_id)
display(stage_df)"""
    ),
]
save(nb07, "07_lora_curriculum_best.ipynb")

# --- 08 eval all tuning models (per-slot cells; see scripts/regenerate_nb08.py) ---
import importlib.util

_reg08 = importlib.util.spec_from_file_location(
    "regenerate_nb08", Path(__file__).resolve().parent / "regenerate_nb08.py"
)
_mod08 = importlib.util.module_from_spec(_reg08)
_reg08.loader.exec_module(_mod08)
_mod08.generate_nb08(root=ROOT)

# --- 09 post-training holdout eval (best_extended + curriculum) ---
nb09 = nbf.v4.new_notebook()
nb09["cells"] = [
    md(
        """# 09 — Holdout evaluation: post-training models (`best_extended` + `curriculum`)

**Summary:** One loop over registry rows with `tuning_stage` in `best_extended` or `curriculum`. Each row writes under `evaluations/holdout_best_extended/<model_id>/` or `evaluations/holdout_curriculum/<model_id>/` and updates `workflow_layout.json` with the matching `eval_kind`. **`MODEL_IDS`:** empty = all such rows; non-empty = filter to those ids **in list order**.

**Prerequisites:** Notebooks **03** (splits), **06** / **07** (trained models in the registry). Run after **08** if you are following the recommended pipeline.

**Disk cache:** Same contract as **08** — `eval_run_manifest.json` per model folder; set `FORCE_REGENERATE_HOLDOUT_PREDICTIONS` to bypass.

**Optional:** Set `SAVE_RENDER_PNGS = True` to write `{id}_target.png` and `{id}_pred.png` under each model's `out_dir/render_pngs/`. Exports use **``sample_df`` only** (up to **``N_SAMPLES_PER_BUCKET``** rows per **``PERCENTILE_BUCKETS``** entry when ``USE_PERCENTILE_BUCKETS`` is true), **not** the full holdout; fallback sample is at most 15 rows."""
    ),
    md("#### Colab / install"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n!pip -q install transformers peft accelerate bitsandbytes cairosvg pillow matplotlib lxml pandas tqdm"),
    md("#### Parameters"),
    code(
        """import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'

MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
MODEL_IDS = []  # empty: all registry rows with tuning_stage in best_extended or curriculum
MAX_NEW_TOKENS_BEST_EXTENDED_DEFAULT = 768
MAX_NEW_TOKENS_CURRICULUM_DEFAULT = 512
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
FORCE_REGENERATE_HOLDOUT_PREDICTIONS = False

# Export PNGs under each model's out_dir/render_pngs/ — sample_df only (N per percentile bucket, not full holdout).
SAVE_RENDER_PNGS = False
RENDER_PNG_SIZE = 256
MAX_RENDER_PNG_ROWS = None  # None = every row of sample_df; int = cap below that"""
    ),
    md("#### Run"),
    code(
        """import json

from src.core.dataframe import choose_first_existing
from src.core.modeling_splits import load_holdout_eval
from src.core.workflow_layout import update_workflow_layout_prediction
from src.eval.holdout_evaluation import (
    enrich_for_display,
    load_holdout_predictions_cached_or_run,
    merge_ranked_metadata,
    sample_percentile_buckets,
)
from src.eval.postprocess_presets import POSTPROCESS_METHODS
from src.training.lora.display import (
    display_cross_model_summary,
    display_prediction_summary,
    display_rendered_comparisons,
    display_text_comparisons,
    save_render_png_pairs,
)
from src.training.lora.registry import load_registry, resolve_adapter_path

if POSTPROCESS_METHOD not in POSTPROCESS_METHODS:
    raise ValueError(f'Unknown POSTPROCESS_METHOD. Pick one of: {sorted(POSTPROCESS_METHODS)}')

holdout = load_holdout_eval(WORKFLOW_ROOT)
PROMPT_COL = choose_first_existing(holdout, ['prompt', 'description', 'text'], 'holdout')
SVG_COL = choose_first_existing(holdout, ['svg', 'svg_code', 'target', 'label'], 'holdout')
reg = load_registry(PROJECT_DIR, models_root=MODELS_ROOT)
mask = reg['tuning_stage'].astype(str).isin(['best_extended', 'curriculum'])
reg_sub = reg[mask].copy()
if len(MODEL_IDS):
    want = {str(x) for x in MODEL_IDS}
    reg_sub = reg_sub[reg_sub['model_id'].astype(str).isin(want)]
    order = {str(m): i for i, m in enumerate(MODEL_IDS)}
    reg_sub['_ord'] = reg_sub['model_id'].astype(str).map(lambda x: order.get(x, 9999))
    reg_sub = reg_sub.sort_values('_ord').drop(columns=['_ord'])
else:
    _stage_order = {'best_extended': 0, 'curriculum': 1}
    reg_sub['_ord'] = reg_sub['tuning_stage'].astype(str).map(lambda x: _stage_order.get(x, 9))
    reg_sub = reg_sub.sort_values(['_ord', 'model_id']).drop(columns=['_ord'])

RANKED_PATH = PROCESSED_DIR / 'train_ranked.csv'

summary_rows = []
for _, rrow in reg_sub.iterrows():
    mid = str(rrow['model_id'])
    ts = str(rrow['tuning_stage'])
    adapter = resolve_adapter_path(PROJECT_DIR, str(rrow['adapter_dir']))
    meta = json.loads(rrow['training_config_json'])
    if ts == 'best_extended':
        eval_root = WORKFLOW_ROOT / 'evaluations' / 'holdout_best_extended'
        eval_kind = 'holdout_best_extended'
        mxt = int(meta.get('max_new_tokens_default', MAX_NEW_TOKENS_BEST_EXTENDED_DEFAULT))
    else:
        eval_root = WORKFLOW_ROOT / 'evaluations' / 'holdout_curriculum'
        eval_kind = 'holdout_curriculum'
        mxt = int(meta.get('max_new_tokens_default', MAX_NEW_TOKENS_CURRICULUM_DEFAULT))
    out_dir = eval_root / mid
    raw_df, _cache_reason = load_holdout_predictions_cached_or_run(
        holdout,
        PROMPT_COL,
        SVG_COL,
        adapter,
        MODEL_ID,
        WORKFLOW_ROOT,
        out_dir,
        POSTPROCESS_METHOD,
        max_new_tokens=mxt,
        force_regenerate=FORCE_REGENERATE_HOLDOUT_PREDICTIONS,
        id_col='id',
    )
    update_workflow_layout_prediction(
        PROJECT_DIR,
        WORKFLOW_ROOT,
        mid,
        out_dir / f'predictions_post_{POSTPROCESS_METHOD}.csv',
        eval_kind=eval_kind,
    )
    disp_df = merge_ranked_metadata(raw_df, RANKED_PATH)
    enriched = enrich_for_display(disp_df, 'pred_svg')
    try:
        sample_df = sample_percentile_buckets(
            enriched, PERCENTILE_BUCKETS, N_SAMPLES_PER_BUCKET, SEED
        )
    except Exception:
        sample_df = enriched.sample(n=min(15, len(enriched)), random_state=SEED)
    if SAVE_RENDER_PNGS:
        paths = save_render_png_pairs(
            sample_df,
            out_dir,
            max_rows=MAX_RENDER_PNG_ROWS,
            output_width=int(RENDER_PNG_SIZE),
            output_height=int(RENDER_PNG_SIZE),
        )
        print(f'Wrote {len(paths) // 2} PNG pair(s) to {out_dir / "render_pngs"}')
    summ = display_prediction_summary(sample_df, heading=f'model_id={mid} ({ts})')
    summary_rows.append({'model_id': mid, 'tuning_stage': ts, **summ})
    display_text_comparisons(sample_df, title=f'Text: {mid}', n_rows=N_TEXT_ROWS, preview_chars=PREVIEW_CHARS)
    display_rendered_comparisons(sample_df, title=f'Render: {mid}', n_rows=N_RENDER_ROWS)

if summary_rows:
    display_cross_model_summary(summary_rows, heading='Post-training models (holdout)')
print('Done post-training holdout eval for', len(summary_rows), 'models')"""
    ),
]
save(nb09, "09_eval_holdout_post_training_models.ipynb")

# --- 10 submission ---
nb11 = nbf.v4.new_notebook()
nb11["cells"] = [
    md(
        """# 10 — Kaggle submission by registry `model_id`

**Summary:** Loads one or more adapters from `<WORKFLOW_ROOT>/models/model_registry.csv`, runs **raw** generation on the **official test CSV** (prompts only), applies **`POSTPROCESS_METHOD`** for submission safety, validates rows, and writes `submission.csv`.

**Parameters:** `RUN_PROFILE_ID` (match notebook **03**), `MODEL_IDS`, `TEST_CSV`, `SUBMISSION_PATH`, `POSTPROCESS_METHOD`, `max_new_tokens` (per model from registry `max_new_tokens_default` when present)."""
    ),
    md("#### Colab / install"),
    code("from google.colab import drive\ndrive.mount('/content/drive')\n!pip -q install transformers peft accelerate bitsandbytes pandas tqdm lxml"),
    md("#### Parameters + run"),
    code(
        """import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RAW_DIR = PROJECT_DIR / 'data' / 'raw'
RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
MODELS_ROOT = WORKFLOW_ROOT / 'models'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'

MODEL_IDS = ['REPLACE_WITH_REGISTRY_MODEL_ID']
BASE_MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
TEST_CSV = RAW_DIR / 'test.csv'
SUBMISSION_PATH = OUTPUTS_DIR / 'submissions' / 'submission.csv'
POSTPROCESS_METHOD = 'current_default_sanitizer'
MAX_NEW_TOKENS_FALLBACK = 768

from src.core.dataframe import choose_first_existing
from src.eval.postprocess_presets import get_postprocess_fn
from src.inference.generation import generate_svg_raw_prediction
from src.svg.cleaning import validate_svg_constraints
from src.training.lora.modeling import load_inference_adapter
from src.training.lora.registry import load_registry, resolve_adapter_path
from src.training.prompts import format_svg_instruction_example

test_df = pd.read_csv(TEST_CSV)
if 'id' not in test_df.columns:
    raise ValueError('test CSV needs id column')
prompt_col = choose_first_existing(test_df, ['prompt', 'description', 'text'], 'test_df')

reg = load_registry(PROJECT_DIR, models_root=MODELS_ROOT)
post_fn = get_postprocess_fn(POSTPROCESS_METHOD)
rows = []
for mid in MODEL_IDS:
    rrow = reg[reg['model_id'].astype(str) == str(mid)]
    if len(rrow) != 1:
        raise ValueError(f'model_id not unique in registry: {mid}')
    rrow = rrow.iloc[0]
    adapter = resolve_adapter_path(PROJECT_DIR, str(rrow['adapter_dir']))
    meta = json.loads(rrow['training_config_json'])
    mxt = int(meta.get('max_new_tokens_default', MAX_NEW_TOKENS_FALLBACK))
    tokenizer, model = load_inference_adapter(adapter, BASE_MODEL_ID)
    for _, trow in tqdm(test_df.iterrows(), total=len(test_df)):
        pid = trow['id']
        prompt_text = str(trow[prompt_col])
        full_prompt = format_svg_instruction_example(prompt_text, svg_text=None, include_answer=False)
        raw = generate_svg_raw_prediction(full_prompt, tokenizer, model, max_new_tokens=mxt)
        svg = post_fn(raw)
        rows.append({'id': pid, 'svg': svg, 'registry_model_id': mid})
    del model, tokenizer

# If multiple MODEL_IDS, last model wins per row; usually pass one id
sub = pd.DataFrame(rows)
if len(MODEL_IDS) > 1:
    sub = sub.drop_duplicates(subset=['id'], keep='last')
sub[['id', 'svg']].to_csv(SUBMISSION_PATH, index=False)
print('Wrote', SUBMISSION_PATH, sub.shape)"""
    ),
]
save(nb11, "10_kaggle_submission_by_model_id.ipynb")

# --- 12 holdout leaderboard ---
nb13 = nbf.v4.new_notebook()
nb13["cells"] = [
    md(
        """# 12 — Holdout evaluation leaderboard (saved table)

**Summary:** For each `MODEL_ID`, loads **postprocessed** holdout predictions produced by notebooks **08** and **09** (first matching eval folder: tuning, best_extended, curriculum), scores **all** holdout rows, joins `eval_loss` from `round1_results.csv`–`round4_results.csv` when `registry_model_id` is present, flattens selected fields from the registry training config, and writes a CSV under `<WORKFLOW_ROOT>/evaluations/holdout_leaderboard/`.

**Prerequisites:** Run **03** (splits) and **08–09** with the same `RUN_PROFILE_ID` so each `model_id` has `predictions_post_<POSTPROCESS_METHOD>.csv` under the profile eval roots."""
    ),
    md("#### Colab / install"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "!pip -q install pandas"
    ),
    md("#### Parameters + run"),
    code(
        """import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path('/content/drive/MyDrive/DL_Midterm_Spring_2026_2/svg_project_DL')
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RUN_PROFILE_ID = 'default'
WORKFLOW_ROOT = PROJECT_DIR / 'outputs' / 'workflow_runs' / RUN_PROFILE_ID
EXPERIMENT_ROOT = WORKFLOW_ROOT / 'lora_tuning_workflow'
LEADERBOARD_DIR = WORKFLOW_ROOT / 'evaluations' / 'holdout_leaderboard'
LEADERBOARD_CSV = LEADERBOARD_DIR / 'holdout_metrics_by_model.csv'

# Registry model_id values to include (same strings as in model_registry.csv)
MODEL_IDS = [
    # 'abc123...',
]
POSTPROCESS_METHOD = 'current_default_sanitizer'

from src.eval.holdout_leaderboard import build_holdout_leaderboard_df

if not MODEL_IDS:
    raise ValueError('Set MODEL_IDS to one or more registry model_id strings.')

df = build_holdout_leaderboard_df(
    PROJECT_DIR,
    WORKFLOW_ROOT,
    MODEL_IDS,
    POSTPROCESS_METHOD,
    workflow_root=WORKFLOW_ROOT,
    experiment_root=EXPERIMENT_ROOT,
)
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(LEADERBOARD_CSV, index=False)
print('Wrote', LEADERBOARD_CSV, df.shape)
display(df)"""
    ),
]
save(nb13, "12_holdout_eval_leaderboard.ipynb")

print("Done 03-12 (plus regenerate 08 eval-all)")
