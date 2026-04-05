import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.core.runtime import cleanup_memory, set_seed, suppress_training_warnings
from src.training.lora.eval import evaluate_generation_panel, summarize_generation_df
from src.training.lora.modeling import apply_lora, load_tokenizer_and_base_model
from src.training.lora.registry import register_model_from_adapter_dir
from src.training.prompts import build_instruction_dataframe


def build_instruction_df(df_in, prompt_col, svg_col):
    """Build an instruction-format DataFrame used for LoRA training.

    Args:
        df_in (pd.DataFrame): Source DataFrame containing prompt and SVG target columns.
        prompt_col (str): Name of the prompt column.
        svg_col (str): Name of the SVG target column.

    Returns:
        pd.DataFrame: Instruction-formatted DataFrame with normalized ``prompt``,
            ``svg_target``, and ``completion`` columns.
    """
    out = build_instruction_dataframe(
        df_in.copy(),
        prompt_col=prompt_col,
        svg_col=svg_col,
    ).reset_index(drop=True)
    out["completion"] = out["svg_target"]
    return out


def build_dataset_from_instruction_df(instruction_df):
    """Convert an instruction DataFrame into a Hugging Face Dataset.

    Args:
        instruction_df (pd.DataFrame): DataFrame containing ``prompt`` and ``completion`` columns.

    Returns:
        datasets.Dataset: Dataset with only the columns required by the SFT trainer.
    """
    return Dataset.from_pandas(
        instruction_df[["prompt", "completion"]].reset_index(drop=True)
    )


def _log_training_device(model) -> None:
    """Print model device information for notebook debugging.

    Args:
        model: Torch model whose parameter placement should be inspected.

    Returns:
        None: Device details and common configuration warnings are printed to stdout.
    """
    cuda_ok = torch.cuda.is_available()
    print(f"[device] torch.cuda.is_available()={cuda_ok}")
    acc = os.environ.get("ACCELERATE_USE_CPU", "")
    if str(acc).lower() in ("1", "true", "yes"):
        print(
            "[device] WARNING: ACCELERATE_USE_CPU is set; training may run on CPU. "
            "For GPU, restart runtime or unset: del os.environ['ACCELERATE_USE_CPU']"
        )
    if cuda_ok:
        print(f"[device] cuda:0 = {torch.cuda.get_device_name(0)}")
    dm = getattr(model, "hf_device_map", None)
    if dm:
        print(f"[device] hf_device_map={dm}")
    else:
        try:
            print(f"[device] first trainable param device={next(model.parameters()).device}")
        except StopIteration:
            pass
    if cuda_ok and not str(acc).lower() in ("1", "true", "yes"):
        try:
            d0 = next(model.parameters()).device
            if d0.type == "cpu":
                print(
                    "[device] WARNING: model reports CPU; 4-bit + bitsandbytes usually needs CUDA. "
                    "If training is very slow, reinstall bitsandbytes for CUDA or check Runtime → GPU."
                )
        except StopIteration:
            pass


def make_training_args(config, output_dir, eval_steps=40, max_length=1024):
    """Construct an ``SFTConfig`` for a LoRA training run.

    Args:
        config (Mapping[str, Any]): Hyperparameter mapping containing keys such as
            ``max_steps``, ``per_device_train_batch_size``, ``gradient_accumulation_steps``,
            ``learning_rate``, ``lora_r``, ``lora_alpha``, and ``lora_dropout``.
        output_dir (str | pathlib.Path): Directory where checkpoints and logs are written.
        eval_steps (int, optional): Evaluation and checkpoint interval in training steps.
            Defaults to ``40``.
        max_length (int, optional): Maximum tokenized sequence length for SFT examples.
            Defaults to ``1024``.

    Returns:
        trl.SFTConfig: Fully configured training-arguments object for ``SFTTrainer``.
    """
    return SFTConfig(
        output_dir=str(output_dir),
        use_cpu=False,
        max_steps=int(config["max_steps"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
        logging_first_step=True,
        remove_unused_columns=False,
        max_length=max_length,
        # Must be explicit: suppress_training_warnings() sets HF log level to ERROR, and
        # TrainingArguments defaults disable_tqdm=True when getEffectiveLevel() > WARN,
        # which hides the training tqdm bar and leaves only dict-style log lines.
        disable_tqdm=False,
    )


def _save_adapter(model, tokenizer, adapter_dir):
    """Save a trained adapter and tokenizer to disk.

    Args:
        model: PEFT model exposing ``save_pretrained``.
        tokenizer: Tokenizer paired with ``model``.
        adapter_dir (str | pathlib.Path): Destination directory for the saved adapter.

    Returns:
        None: Files are written into ``adapter_dir``.
    """
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))


class _PanelMetricCallback(TrainerCallback):
    """Track fixed-panel generation metrics and save the best adapter checkpoint.

    Args:
        tokenizer: Tokenizer used to decode evaluation generations.
        val_instruction_df (pd.DataFrame): Validation instruction DataFrame used for panel
            generation.
        run_dir (str | pathlib.Path): Run directory where best adapters are saved.
        panel_n (int, optional): Number of validation examples evaluated per panel. Defaults to
            ``12``.
        panel_seed (int, optional): Random seed used when sampling the panel. Defaults to ``42``.
        metric_name (str, optional): Summary metric key to optimize, such as ``"render_rate"``.
            Defaults to ``"render_rate"``.
        patience (int | None, optional): Number of non-improving evaluations to tolerate before
            early stopping. ``None`` disables callback-driven stopping. Defaults to ``1``.
        min_delta (float, optional): Minimum metric improvement required to reset patience.
            Defaults to ``0.0``.

    Attributes:
        history_rows (list[dict]): Per-evaluation metric history.
        best_metric (float): Best observed task metric value.
        best_generation_df (pd.DataFrame): Panel DataFrame associated with the best metric.
        best_adapter_dir (pathlib.Path): Directory where the best adapter snapshot is stored.
    """

    def __init__(
        self,
        tokenizer,
        val_instruction_df,
        run_dir,
        panel_n=12,
        panel_seed=42,
        metric_name="render_rate",
        patience=1,
        min_delta=0.0,
    ):
        """Initialize panel-metric tracking state.

        Args:
            tokenizer: Tokenizer used for generation during evaluation.
            val_instruction_df (pd.DataFrame): Validation examples used for panel scoring.
            run_dir (str | pathlib.Path): Directory where best-adapter artifacts are saved.
            panel_n (int, optional): Number of examples to evaluate each time. Defaults to ``12``.
            panel_seed (int, optional): Random seed for panel sampling. Defaults to ``42``.
            metric_name (str, optional): Summary metric used to choose the best adapter.
            patience (int | None, optional): Maximum consecutive non-improving evaluations before
                early stopping. ``None`` disables callback-controlled stopping.
            min_delta (float, optional): Minimum metric improvement treated as progress.

        Returns:
            None: Tracking fields are stored on ``self``.
        """
        self.tokenizer = tokenizer
        self.val_instruction_df = val_instruction_df
        self.run_dir = Path(run_dir)
        self.panel_n = int(panel_n)
        self.panel_seed = int(panel_seed)
        self.metric_name = str(metric_name)
        self.patience = None if patience is None else max(1, int(patience))
        self.use_early_stopping = self.patience is not None
        self.min_delta = float(min_delta)
        self.history_rows = []
        self.best_metric = -np.inf
        self.best_step = None
        self.best_generation_df = pd.DataFrame()
        self.bad_eval_count = 0
        self.stop_reason = ""
        self.best_adapter_dir = self.run_dir / "best_task_adapter"

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Evaluate a fixed panel after each validation event.

        Args:
            args: Trainer arguments supplied by Hugging Face.
            state: Trainer state containing the current global step.
            control: Trainer control object that can request early stopping.
            model: Model being evaluated. If ``None``, no panel is computed.
            metrics (dict | None, optional): Evaluation metrics reported by the trainer.
            **kwargs: Additional callback arguments unused by this implementation.

        Returns:
            transformers.TrainerControl: Possibly updated control object. ``should_training_stop``
                may be set when the monitored panel metric stops improving.
        """
        if model is None or self.panel_n <= 0:
            return control
        generation_df = evaluate_generation_panel(
            model,
            self.tokenizer,
            self.val_instruction_df,
            n_examples=self.panel_n,
            seed=self.panel_seed,
        )
        panel_summary = summarize_generation_df(generation_df)
        metric_value = float(panel_summary.get(self.metric_name, np.nan))
        improved = np.isfinite(metric_value) and (
            metric_value > (self.best_metric + self.min_delta)
        )

        history_row = {
            "global_step": int(state.global_step),
            "eval_loss": float(metrics.get("eval_loss")) if metrics and metrics.get("eval_loss") is not None else np.nan,
            "task_metric_name": self.metric_name,
            "task_metric_value": metric_value,
            "task_metric_improved": bool(improved),
            **panel_summary,
        }

        if improved:
            self.best_metric = metric_value
            self.best_step = int(state.global_step)
            self.best_generation_df = generation_df.copy()
            self.best_generation_df["eval_step"] = int(state.global_step)
            if self.best_adapter_dir.exists():
                shutil.rmtree(self.best_adapter_dir)
            _save_adapter(model, self.tokenizer, self.best_adapter_dir)
            self.bad_eval_count = 0
        else:
            self.bad_eval_count += 1

        history_row["task_metric_bad_eval_count"] = int(self.bad_eval_count)
        self.history_rows.append(history_row)

        if self.use_early_stopping and self.bad_eval_count >= self.patience:
            self.stop_reason = (
                f"{self.metric_name} did not improve for {self.bad_eval_count} evaluation(s)"
            )
            control.should_training_stop = True
        return control


def run_single_experiment(
    run_name,
    config,
    train_df,
    val_df,
    prompt_col,
    svg_col,
    model_id,
    max_seq_length,
    root_dir,
    panel_n=0,
    seed=42,
):
    """Run one baseline LoRA fine-tuning experiment.

    Args:
        run_name (str): Name of the experiment subdirectory created under ``root_dir``.
        config (Mapping[str, Any]): Hyperparameter mapping consumed by ``make_training_args`` and
            ``apply_lora``.
        train_df (pd.DataFrame): Training split containing prompt/SVG columns.
        val_df (pd.DataFrame): Validation split containing prompt/SVG columns.
        prompt_col (str): Prompt column name shared by ``train_df`` and ``val_df``.
        svg_col (str): SVG target column name shared by ``train_df`` and ``val_df``.
        model_id (str): Base model identifier to load before applying LoRA.
        max_seq_length (int): Maximum sequence length used by the SFT trainer.
        root_dir (str | pathlib.Path): Directory where the run folder is created.
        panel_n (int, optional): Number of validation examples to generate for the optional
            post-training panel. ``0`` skips panel generation. Defaults to ``0``.
        seed (int, optional): Random seed for sampling and training reproducibility.
            Defaults to ``42``.

    Returns:
        tuple[dict, pd.DataFrame, pathlib.Path]: ``(summary, generation_df, run_dir)`` where the
            summary is also saved to ``summary.csv`` and the adapter is saved under
            ``run_dir / "adapter"``.
    """
    set_seed(seed)
    suppress_training_warnings()
    cleanup_memory()
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_instruction_df = build_instruction_df(train_df, prompt_col=prompt_col, svg_col=svg_col)
    val_instruction_df = build_instruction_df(val_df, prompt_col=prompt_col, svg_col=svg_col)
    if "difficulty_bucket" in train_df.columns:
        train_instruction_df["difficulty_bucket"] = train_df["difficulty_bucket"].values
    if "difficulty_bucket" in val_df.columns:
        val_instruction_df["difficulty_bucket"] = val_df["difficulty_bucket"].values
    train_dataset = build_dataset_from_instruction_df(train_instruction_df)
    eval_dataset = build_dataset_from_instruction_df(val_instruction_df)
    tokenizer, model = load_tokenizer_and_base_model(model_id)
    model = apply_lora(model, config)
    _log_training_device(model)
    trainer = SFTTrainer(
        model=model,
        args=make_training_args(
            config,
            output_dir=run_dir,
            eval_steps=max(20, int(config["max_steps"]) // 3),
            max_length=max_seq_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    # Optional val-set generation panel (slow). Default 0: skip; use eval notebooks 07–09 for holdout eval.
    if panel_n and panel_n > 0:
        generation_df = evaluate_generation_panel(
            model, tokenizer, val_instruction_df, n_examples=panel_n, seed=seed
        )
    else:
        generation_df = pd.DataFrame()
    gen_summary = summarize_generation_df(generation_df)
    summary = {
        "run_name": run_name,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "learning_rate": float(config["learning_rate"]),
        "lora_r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_batch_size": int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]),
        "max_steps": int(config["max_steps"]),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "eval_loss": eval_metrics.get("eval_loss"),
        **gen_summary,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    generation_df.to_csv(run_dir / "generation_panel.csv", index=False)
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    del trainer
    del model
    del tokenizer
    cleanup_memory()
    return summary, generation_df, run_dir


def run_single_experiment_v2(
    run_name,
    config,
    train_df,
    val_df,
    prompt_col,
    svg_col,
    model_id,
    max_seq_length,
    root_dir,
    eval_steps=50,
    panel_n=12,
    seed=42,
    task_metric_name="render_rate",
    use_task_metric_early_stopping=True,
    early_stopping_patience=1,
    early_stopping_min_delta=0.0,
):
    """Run an enhanced LoRA experiment with task-metric tracking.

    Args:
        run_name (str): Name of the run directory created under ``root_dir``.
        config (Mapping[str, Any]): Hyperparameter mapping for LoRA and SFT setup.
        train_df (pd.DataFrame): Training split with prompt/SVG columns.
        val_df (pd.DataFrame): Validation split with prompt/SVG columns.
        prompt_col (str): Prompt column name in both splits.
        svg_col (str): SVG target column name in both splits.
        model_id (str): Base model identifier to fine-tune.
        max_seq_length (int): Maximum sequence length supplied to the trainer.
        root_dir (str | pathlib.Path): Parent directory where run outputs are written.
        eval_steps (int, optional): Validation frequency in training steps. Defaults to ``50``.
        panel_n (int, optional): Number of examples used in the fixed evaluation panel.
            Defaults to ``12``.
        seed (int, optional): Random seed for reproducibility. Defaults to ``42``.
        task_metric_name (str, optional): Panel summary metric used for model selection.
            Defaults to ``"render_rate"``.
        use_task_metric_early_stopping (bool, optional): If ``True``, stop when the task metric
            stops improving. Defaults to ``True``.
        early_stopping_patience (int, optional): Number of non-improving evaluations tolerated
            before stopping. Defaults to ``1``.
        early_stopping_min_delta (float, optional): Minimum metric improvement required to count
            as progress. Defaults to ``0.0``.

    Returns:
        tuple[dict, pd.DataFrame, pathlib.Path, pd.DataFrame]: ``(summary, generation_df,
        run_dir, eval_history_df)`` with CSV artifacts written into ``run_dir``.
    """
    set_seed(seed)
    suppress_training_warnings()
    cleanup_memory()
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_instruction_df = build_instruction_df(train_df, prompt_col=prompt_col, svg_col=svg_col)
    val_instruction_df = build_instruction_df(val_df, prompt_col=prompt_col, svg_col=svg_col)
    if "difficulty_bucket" in train_df.columns:
        train_instruction_df["difficulty_bucket"] = train_df["difficulty_bucket"].values
    if "difficulty_bucket" in val_df.columns:
        val_instruction_df["difficulty_bucket"] = val_df["difficulty_bucket"].values
    train_dataset = build_dataset_from_instruction_df(train_instruction_df)
    eval_dataset = build_dataset_from_instruction_df(val_instruction_df)
    tokenizer, model = load_tokenizer_and_base_model(model_id)
    model = apply_lora(model, config)
    _log_training_device(model)

    panel_callback = _PanelMetricCallback(
        tokenizer=tokenizer,
        val_instruction_df=val_instruction_df,
        run_dir=run_dir,
        panel_n=panel_n,
        panel_seed=seed,
        metric_name=task_metric_name,
        patience=early_stopping_patience if use_task_metric_early_stopping else None,
        min_delta=early_stopping_min_delta,
    )

    trainer = SFTTrainer(
        model=model,
        args=make_training_args(
            config,
            output_dir=run_dir,
            eval_steps=eval_steps,
            max_length=max_seq_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=max(1, int(early_stopping_patience)),
                early_stopping_threshold=float(early_stopping_min_delta),
            ),
            panel_callback,
        ],
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    eval_history_df = pd.DataFrame(panel_callback.history_rows)
    eval_history_df.to_csv(run_dir / "eval_history.csv", index=False)

    if len(panel_callback.best_generation_df):
        generation_df = panel_callback.best_generation_df.copy()
        generation_df.to_csv(run_dir / "generation_panel_best.csv", index=False)
        adapter_dir = run_dir / "adapter"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(str(panel_callback.best_adapter_dir), str(adapter_dir))
    else:
        generation_df = pd.DataFrame()
        generation_df.to_csv(run_dir / "generation_panel_best.csv", index=False)
        _save_adapter(model, tokenizer, run_dir / "adapter")

    gen_summary = summarize_generation_df(generation_df)
    stop_reason = panel_callback.stop_reason
    if not stop_reason and trainer.state.global_step < int(config["max_steps"]):
        stop_reason = "eval_loss did not improve enough for early stopping"
    if not stop_reason:
        stop_reason = "max_steps reached"
    summary = {
        "run_name": run_name,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "learning_rate": float(config["learning_rate"]),
        "lora_r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_batch_size": int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]),
        "max_steps": int(config["max_steps"]),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "eval_steps": int(eval_steps),
        "task_metric_name": str(task_metric_name),
        "use_task_metric_early_stopping": bool(use_task_metric_early_stopping),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "final_eval_loss": eval_metrics.get("eval_loss"),
        "best_loss_checkpoint": trainer.state.best_model_checkpoint,
        "best_task_metric_step": panel_callback.best_step,
        "best_task_metric_value": float(panel_callback.best_metric) if np.isfinite(panel_callback.best_metric) else np.nan,
        "stop_reason": stop_reason,
        **gen_summary,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    generation_df.to_csv(run_dir / "generation_panel.csv", index=False)
    del trainer
    del model
    del tokenizer
    cleanup_memory()
    return summary, generation_df, run_dir, eval_history_df


def run_curriculum_experiment(
    run_name,
    config,
    reduced_train_df,
    val_df,
    prompt_col,
    svg_col,
    model_id,
    max_seq_length,
    root_dir,
    stage_fracs,
    stage_labels,
    stage_max_steps=None,
    eval_steps=50,
    panel_n=12,
    seed=42,
    task_metric_name="render_rate",
    use_task_metric_early_stopping=True,
    early_stopping_patience=1,
    early_stopping_min_delta=0.0,
):
    """Run multi-stage curriculum fine-tuning from easier to harder subsets.

    Args:
        run_name (str): Name of the curriculum run directory.
        config (Mapping[str, Any]): Base hyperparameter mapping shared across stages.
        reduced_train_df (pd.DataFrame): Training DataFrame already sorted or filtered for the
            intended curriculum order.
        val_df (pd.DataFrame): Validation split used for stage evaluation.
        prompt_col (str): Prompt column name.
        svg_col (str): SVG target column name.
        model_id (str): Base model identifier to fine-tune.
        max_seq_length (int): Maximum sequence length supplied to the trainer.
        root_dir (str | pathlib.Path): Parent directory where stage artifacts are written.
        stage_fracs (Sequence[float]): Fractions of ``reduced_train_df`` used at each stage.
        stage_labels (Sequence[str]): Human-readable stage names used as directory names.
        stage_max_steps (Sequence[int] | None, optional): Per-stage maximum step counts. ``None``
            reuses ``config["max_steps"]`` for every stage.
        eval_steps (int, optional): Validation frequency in steps. Defaults to ``50``.
        panel_n (int, optional): Number of validation examples used for the final generation
            panel. Defaults to ``12``.
        seed (int, optional): Random seed for reproducibility. Defaults to ``42``.
        task_metric_name (str, optional): Recorded task metric name included in summary exports.
            Defaults to ``"render_rate"``.
        use_task_metric_early_stopping (bool, optional): Recorded in outputs for parity with
            non-curriculum experiments. Defaults to ``True``.
        early_stopping_patience (int, optional): Hugging Face early-stopping patience per stage.
            Defaults to ``1``.
        early_stopping_min_delta (float, optional): Minimum eval-loss improvement threshold for
            stage early stopping. Defaults to ``0.0``.

    Returns:
        tuple[dict, pd.DataFrame, pathlib.Path, pd.DataFrame]: ``(summary, generation_df,
        run_dir, stage_df)`` where ``stage_df`` contains one row per curriculum stage.
    """
    set_seed(seed)
    suppress_training_warnings()
    cleanup_memory()
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    reduced_train_df = reduced_train_df.reset_index(drop=True).copy()
    if stage_max_steps is None:
        stage_max_steps = [int(config["max_steps"])] * len(stage_fracs)
    stage_max_steps = [int(v) for v in stage_max_steps]
    if len(stage_max_steps) != len(stage_fracs):
        raise ValueError(
            f"stage_max_steps must have length {len(stage_fracs)} but got {len(stage_max_steps)}"
        )
    val_instruction_df = build_instruction_df(val_df, prompt_col=prompt_col, svg_col=svg_col)
    if "difficulty_bucket" in val_df.columns:
        val_instruction_df["difficulty_bucket"] = val_df["difficulty_bucket"].values
    eval_dataset = build_dataset_from_instruction_df(val_instruction_df)
    tokenizer, model = load_tokenizer_and_base_model(model_id)
    model = apply_lora(model, config)
    _log_training_device(model)
    stage_summaries = []
    best_stage_eval_loss = np.inf
    best_stage_label = ""
    best_loss_checkpoint = ""
    stop_reason = ""
    for stage_idx, frac in enumerate(stage_fracs):
        stage_label = stage_labels[stage_idx] if stage_idx < len(stage_labels) else f"stage_{stage_idx + 1}"
        stage_max_steps_value = int(stage_max_steps[stage_idx])
        stage_n = max(1, int(round(len(reduced_train_df) * frac)))
        stage_train_df = reduced_train_df.iloc[:stage_n].copy().reset_index(drop=True)
        stage_instruction_df = build_instruction_df(stage_train_df, prompt_col=prompt_col, svg_col=svg_col)
        if "difficulty_bucket" in stage_train_df.columns:
            stage_instruction_df["difficulty_bucket"] = stage_train_df["difficulty_bucket"].values
        stage_dataset = build_dataset_from_instruction_df(stage_instruction_df)
        stage_output_dir = run_dir / stage_label
        stage_output_dir.mkdir(parents=True, exist_ok=True)
        stage_config = dict(config)
        stage_config["max_steps"] = stage_max_steps_value
        trainer = SFTTrainer(
            model=model,
            args=make_training_args(
                stage_config,
                output_dir=stage_output_dir,
                eval_steps=eval_steps,
                max_length=max_seq_length,
            ),
            train_dataset=stage_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=max(1, int(early_stopping_patience)),
                    early_stopping_threshold=float(early_stopping_min_delta),
                )
            ],
        )
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        stage_eval_loss = eval_metrics.get("eval_loss")
        if stage_eval_loss is not None and np.isfinite(stage_eval_loss) and stage_eval_loss < best_stage_eval_loss:
            best_stage_eval_loss = float(stage_eval_loss)
            best_stage_label = stage_label
            best_loss_checkpoint = str(trainer.state.best_model_checkpoint or "")
        stage_stop_reason = "max_steps reached"
        if trainer.state.global_step < stage_max_steps_value:
            stage_stop_reason = "eval_loss did not improve enough for early stopping"
        stop_reason = stage_stop_reason

        # Export the adapter after each curriculum stage so downstream notebooks
        # can load a specific stage for qualitative inspection.
        stage_adapter_dir = stage_output_dir / "adapter"
        stage_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(stage_adapter_dir))
        tokenizer.save_pretrained(str(stage_adapter_dir))

        stage_summaries.append(
            {
                "stage_idx": stage_idx + 1,
                "stage_label": stage_label,
                "stage_frac": frac,
                "stage_rows": len(stage_train_df),
                "stage_max_steps": stage_max_steps_value,
                "eval_steps": int(eval_steps),
                "task_metric_name": str(task_metric_name),
                "use_task_metric_early_stopping": bool(use_task_metric_early_stopping),
                "early_stopping_patience": int(early_stopping_patience),
                "early_stopping_min_delta": float(early_stopping_min_delta),
                "train_loss": train_result.metrics.get("train_loss"),
                "train_runtime": train_result.metrics.get("train_runtime"),
                "eval_loss": eval_metrics.get("eval_loss"),
                "best_loss_checkpoint": trainer.state.best_model_checkpoint,
                "stop_reason": stage_stop_reason,
            }
        )
        del trainer
        cleanup_memory()
    if panel_n and panel_n > 0:
        generation_df = evaluate_generation_panel(
            model, tokenizer, val_instruction_df, n_examples=panel_n, seed=seed
        )
    else:
        generation_df = pd.DataFrame()
    gen_summary = summarize_generation_df(generation_df)
    final_eval_metrics = {"eval_loss": np.nan}
    stage_df = pd.DataFrame(stage_summaries)
    if len(stage_df):
        final_eval_metrics["eval_loss"] = stage_df.iloc[-1]["eval_loss"]
    summary = {
        "run_name": run_name,
        "train_rows": len(reduced_train_df),
        "val_rows": len(val_df),
        "learning_rate": float(config["learning_rate"]),
        "lora_r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_batch_size": int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]),
        "max_steps": int(config["max_steps"]),
        "stage_max_steps": ",".join(str(v) for v in stage_max_steps),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "eval_steps": int(eval_steps),
        "task_metric_name": str(task_metric_name),
        "use_task_metric_early_stopping": bool(use_task_metric_early_stopping),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "eval_loss": final_eval_metrics.get("eval_loss"),
        "best_stage_label": best_stage_label,
        "best_loss_checkpoint": best_loss_checkpoint,
        "stop_reason": stop_reason or "max_steps reached",
        **gen_summary,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    stage_df.to_csv(run_dir / "curriculum_stage_summary.csv", index=False)
    generation_df.to_csv(run_dir / "generation_panel.csv", index=False)
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    del model
    del tokenizer
    cleanup_memory()
    return summary, generation_df, run_dir, stage_df


def run_single_experiment_eval_loss_early_stop(
    run_name,
    config,
    train_df,
    val_df,
    prompt_col,
    svg_col,
    model_id,
    max_seq_length,
    root_dir,
    project_root,
    tuning_stage,
    curriculum=False,
    seed=42,
    eval_steps=None,
    notes="",
    registry_extra=None,
    models_root=None,
):
    """LoRA SFT with ``EarlyStoppingCallback`` on ``eval_loss`` only (no generation panel).

    Saves checkpoints under ``root_dir / run_name``, then copies the trained adapter to
    ``<models_root>/lora_model_id_<id>/`` (default ``outputs/models``) and appends registry CSV.

    Args:
        run_name: Subdirectory name under ``root_dir``.
        config: Hyperparameters including ``max_steps``, LoRA fields, batch sizes, optional
            ``early_stopping_patience`` (default 3), ``early_stopping_min_delta`` (default 0),
            and optional ``eval_steps``.
        project_root: Repository root (parent of ``outputs/``).
        tuning_stage: Registry label, e.g. ``round1``, ``round4``, ``best_extended``.
        curriculum: Stored in registry (usually ``False`` for non-curriculum runs).
        eval_steps: Override validation frequency; default derived from ``max_steps`` if omitted.
        registry_extra: Optional dict merged into ``training_config_json`` before CSV append
            (e.g. ``max_new_tokens_default`` for inference defaults).
        models_root: Optional directory for registered adapters and ``model_registry.csv``.

    Returns:
        ``(summary, generation_df, run_dir, registry_model_id)`` where ``generation_df`` is empty.
    """
    set_seed(seed)
    suppress_training_warnings()
    cleanup_memory()
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_instruction_df = build_instruction_df(train_df, prompt_col=prompt_col, svg_col=svg_col)
    val_instruction_df = build_instruction_df(val_df, prompt_col=prompt_col, svg_col=svg_col)
    if "difficulty_bucket" in train_df.columns:
        train_instruction_df["difficulty_bucket"] = train_df["difficulty_bucket"].values
    if "difficulty_bucket" in val_df.columns:
        val_instruction_df["difficulty_bucket"] = val_df["difficulty_bucket"].values
    train_dataset = build_dataset_from_instruction_df(train_instruction_df)
    eval_dataset = build_dataset_from_instruction_df(val_instruction_df)
    tokenizer, model = load_tokenizer_and_base_model(model_id)
    model = apply_lora(model, config)
    _log_training_device(model)

    if eval_steps is None:
        eval_steps = config.get("eval_steps")
    if eval_steps is None:
        eval_steps = max(20, int(config["max_steps"]) // 3)
    eval_steps = int(eval_steps)
    early_stopping_patience = int(config.get("early_stopping_patience", 3))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))

    trainer = SFTTrainer(
        model=model,
        args=make_training_args(
            config,
            output_dir=run_dir,
            eval_steps=eval_steps,
            max_length=max_seq_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=max(1, early_stopping_patience),
                early_stopping_threshold=early_stopping_min_delta,
            ),
        ],
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    generation_df = pd.DataFrame()
    gen_summary = summarize_generation_df(generation_df)

    training_config = {
        "base_model_id": model_id,
        "max_seq_length": int(max_seq_length),
        "tuning_stage": tuning_stage,
        "curriculum": bool(curriculum),
        "run_name": run_name,
        "eval_steps": eval_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        **{k: v for k, v in config.items()},
    }
    if registry_extra:
        training_config.update(dict(registry_extra))

    summary = {
        "run_name": run_name,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "learning_rate": float(config["learning_rate"]),
        "lora_r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "effective_batch_size": int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"]),
        "max_steps": int(config["max_steps"]),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "best_loss_checkpoint": trainer.state.best_model_checkpoint,
        **gen_summary,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
    generation_df.to_csv(run_dir / "generation_panel.csv", index=False)
    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))

    registry_model_id = register_model_from_adapter_dir(
        project_root,
        run_dir / "adapter",
        curriculum=curriculum,
        tuning_stage=tuning_stage,
        training_config=training_config,
        notes=notes,
        models_root=models_root,
    )
    summary["registry_model_id"] = registry_model_id
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    del trainer
    del model
    del tokenizer
    cleanup_memory()
    return summary, generation_df, run_dir, registry_model_id


def run_curriculum_experiment_eval_loss_only(
    run_name,
    config,
    reduced_train_df,
    val_df,
    prompt_col,
    svg_col,
    model_id,
    max_seq_length,
    root_dir,
    project_root,
    stage_fracs,
    stage_labels,
    stage_max_steps=None,
    eval_steps=50,
    seed=42,
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,
    notes="",
    models_root=None,
):
    """Curriculum LoRA training with eval-loss early stopping only; register final adapter.

    Does not use task-metric callbacks. Sets ``panel_n=0`` to skip slow generation panels.

    Returns:
        ``(summary, generation_df, run_dir, stage_df, registry_model_id)``
    """
    summary, generation_df, run_dir, stage_df = run_curriculum_experiment(
        run_name=run_name,
        config=config,
        reduced_train_df=reduced_train_df,
        val_df=val_df,
        prompt_col=prompt_col,
        svg_col=svg_col,
        model_id=model_id,
        max_seq_length=max_seq_length,
        root_dir=root_dir,
        stage_fracs=stage_fracs,
        stage_labels=stage_labels,
        stage_max_steps=stage_max_steps,
        eval_steps=eval_steps,
        panel_n=0,
        seed=seed,
        task_metric_name="render_rate",
        use_task_metric_early_stopping=False,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    stage_list = (
        [int(x) for x in stage_max_steps]
        if stage_max_steps is not None
        else [int(config["max_steps"])] * len(stage_fracs)
    )
    training_config = {
        "base_model_id": model_id,
        "max_seq_length": int(max_seq_length),
        "tuning_stage": "curriculum",
        "curriculum": True,
        "stage_fracs": [float(x) for x in stage_fracs],
        "stage_labels": list(stage_labels),
        "stage_max_steps": stage_list,
        "eval_steps": int(eval_steps),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        **{k: v for k, v in config.items()},
    }

    registry_model_id = register_model_from_adapter_dir(
        project_root,
        run_dir / "adapter",
        curriculum=True,
        tuning_stage="curriculum",
        training_config=training_config,
        notes=notes,
        models_root=models_root,
    )
    summary["registry_model_id"] = registry_model_id
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    return summary, generation_df, run_dir, stage_df, registry_model_id
