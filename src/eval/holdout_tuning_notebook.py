"""Shared helpers for notebook 08 (holdout eval over many registry models).

Keeps generation and display logic in one importable module so the notebook stays a thin shell
of parameters and per-model cells (similar to legacy notebook 09).
"""

from __future__ import annotations

import html as html_module
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from IPython.display import HTML, display

from src.core.dataframe import choose_first_existing
from src.core.modeling_splits import load_holdout_eval
from src.core.workflow_layout import update_workflow_layout_prediction
from src.core.runtime import cleanup_memory
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
)
from src.training.lora.eval import summarize_generation_df
from src.training.lora.registry import load_registry, resolve_adapter_path


def build_holdout_models_list(
    project_root: Path,
    outputs_dir: Path | None = None,
    *,
    workflow_root: Path | None = None,
    tuning_stages: set[str] | None = None,
    model_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return ordered registry rows for holdout evaluation (dicts with model_id, adapter path, etc.).

    If ``model_ids`` is non-empty, keep only those ids, in the **order given** (skips unknown ids).
    """
    if tuning_stages is None:
        tuning_stages = {"round1", "round2", "round3", "round4"}
    models_root = (workflow_root / "models") if workflow_root is not None else None
    reg = load_registry(project_root, models_root=models_root)
    reg_sub = reg[reg["tuning_stage"].isin(tuning_stages)].copy()
    reg_sub = reg_sub.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, r in reg_sub.iterrows():
        rows.append(
            {
                "model_id": str(r["model_id"]),
                "tuning_stage": str(r["tuning_stage"]),
                "adapter_dir": resolve_adapter_path(project_root, str(r["adapter_dir"])),
                "registry_row": r,
            }
        )
    if model_ids is not None and len(model_ids) > 0:
        by_id = {d["model_id"]: d for d in rows}
        rows = []
        for mid in model_ids:
            sm = str(mid)
            if sm in by_id:
                rows.append(by_id[sm])
    for i, d in enumerate(rows):
        d["model_index"] = i + 1
    return rows


def prepare_holdout_and_columns(outputs_dir: Path):
    """Load holdout CSV; return (holdout_df, prompt_col, svg_col)."""
    holdout = load_holdout_eval(outputs_dir)
    prompt_col = choose_first_existing(holdout, ["prompt", "description", "text"], "holdout")
    svg_col = choose_first_existing(holdout, ["svg", "svg_code", "target", "label"], "holdout")
    return holdout, prompt_col, svg_col


class HoldoutTuningSession:
    """Caches predictions per model index in memory for display (legacy-style).

    Disk reuse of ``predictions_raw.csv`` is controlled by ``eval_run_manifest.json`` and
    ``force_regenerate``; ``clear()`` only empties this session's RAM cache.
    """

    def __init__(self) -> None:
        self.predictions_by_index: dict[int, dict[str, Any]] = {}
        self.postprocessed_summary_rows: list[dict] = []

    def clear(self) -> None:
        """Drop in-memory display cache only; does not delete on-disk predictions."""
        self.predictions_by_index.clear()
        self.postprocessed_summary_rows.clear()

    def run_model_inference(
        self,
        model_index: int,
        *,
        holdout_models: list[dict],
        holdout: pd.DataFrame,
        prompt_col: str,
        svg_col: str,
        base_model_id: str,
        max_new_tokens: int,
        postprocess_method: str,
        eval_root: Path,
        ranked_path: Path,
        use_percentile_buckets: bool,
        percentile_buckets: list,
        n_samples_per_bucket: int,
        seed: int,
        show_raw_metrics: bool,
        outputs_dir: Path,
        force_regenerate: bool = False,
        head_n: int = 15,
        project_root: Path | None = None,
        workflow_root: Path | None = None,
        eval_kind: str = "holdout_tuning",
    ) -> None:
        if model_index < 1 or model_index > len(holdout_models):
            print(
                f"[skip] No model at index {model_index} "
                f"(registry has {len(holdout_models)} model(s) for this stage filter)."
            )
            return

        if postprocess_method not in POSTPROCESS_METHODS:
            raise ValueError(
                f"Unknown postprocess_method {postprocess_method!r}. "
                f"Choose one of: {sorted(POSTPROCESS_METHODS)}"
            )

        rec = holdout_models[model_index - 1]
        mid = rec["model_id"]
        adapter = rec["adapter_dir"]
        stage = rec["tuning_stage"]
        out_dir = eval_root / mid

        display(
            HTML(
                '<hr style="margin:1.5em 0 0.75em 0;border:none;border-top:2px solid #444;">'
                f'<h2 style="margin:0;">Model index {model_index}: <code>{html_module.escape(mid)}</code></h2>'
                f'<p style="margin:0.25em 0 0 0;"><b>tuning_stage:</b> {html_module.escape(stage)}</p>'
                f'<p style="margin:0;font-size:12px;"><code>{html_module.escape(str(adapter))}</code></p>'
            )
        )

        raw_df, cache_reason = load_holdout_predictions_cached_or_run(
            holdout,
            prompt_col,
            svg_col,
            adapter,
            base_model_id,
            outputs_dir,
            out_dir,
            postprocess_method,
            max_new_tokens=max_new_tokens,
            force_regenerate=force_regenerate,
            id_col="id",
        )
        if project_root is not None and workflow_root is not None:
            post_csv = out_dir / f"predictions_post_{postprocess_method}.csv"
            update_workflow_layout_prediction(
                project_root,
                workflow_root,
                mid,
                post_csv,
                eval_kind=eval_kind,
            )
        display(
            HTML(
                f"<p><i>Holdout predictions:</i> <b>{html_module.escape(cache_reason)}</b> "
                f"(see <code>eval_run_manifest.json</code> under this model folder).</p>"
            )
        )
        disp_df = merge_ranked_metadata(raw_df, ranked_path)
        enriched = enrich_for_display(disp_df, "pred_svg")

        if use_percentile_buckets:
            try:
                sample_df = sample_percentile_buckets(
                    enriched, percentile_buckets, n_samples_per_bucket, seed
                )
            except Exception:
                sample_df = enriched.sample(n=min(head_n, len(enriched)), random_state=seed)
        else:
            sample_df = enriched.head(head_n)

        self.predictions_by_index[model_index] = {
            "meta": rec,
            "sample_df": sample_df,
            "out_dir": out_dir,
            "show_raw_metrics": show_raw_metrics,
            "postprocess_method": postprocess_method,
        }

        summ = summarize_generation_df(sample_df)
        self.postprocessed_summary_rows = [
            r for r in self.postprocessed_summary_rows if r.get("model_index") != model_index
        ]
        self.postprocessed_summary_rows.append(
            {
                "model_index": model_index,
                "model_id": mid,
                "tuning_stage": stage,
                **summ,
            }
        )

        del raw_df
        cleanup_memory()
        display(
            HTML(
                f"<p><b>Done.</b> Model index {model_index} — session ready for display cells. "
                f"Artifacts under <code>{html_module.escape(str(out_dir))}</code></p>"
            )
        )
        self.refresh_summary_table()

    def refresh_summary_table(self) -> None:
        if not self.postprocessed_summary_rows:
            display(HTML("<p><i>No models in summary yet.</i></p>"))
            return
        rows = sorted(self.postprocessed_summary_rows, key=lambda r: r["model_index"])
        display_cross_model_summary(rows, heading="Cross-model summary (postprocessed metrics, cached runs)")

    def run_all_models_inference(self, *, holdout_models: list, **kwargs: Any) -> None:
        for k in range(1, len(holdout_models) + 1):
            self.run_model_inference(k, holdout_models=holdout_models, **kwargs)

    def _get_payload(self, model_index: int) -> dict[str, Any]:
        if model_index not in self.predictions_by_index:
            raise ValueError(
                f"No cached predictions for model index {model_index}. "
                f"Run `SESSION.run_model_inference({model_index}, ...)` first "
                f"(or `run_all_models_inference`)."
            )
        return self.predictions_by_index[model_index]

    def display_raw_text(self, model_index: int, *, n_rows: int | None, preview_chars: int) -> None:
        p = self._get_payload(model_index)
        sample_df = p["sample_df"]
        display_text_comparisons(
            sample_df,
            title=f"Model {model_index} — text: target vs raw output",
            subtitle="Left = holdout ground truth; right = model decode before postprocessing.",
            n_rows=n_rows,
            preview_chars=preview_chars,
            pred_col="raw_pred",
            right_heading="Raw model output (text)",
        )

    def display_raw_rendered(self, model_index: int, *, n_rows: int | None) -> None:
        p = self._get_payload(model_index)
        sample_df = p["sample_df"]
        display_rendered_comparisons(
            sample_df,
            title=f"Model {model_index} — rendered: target vs raw output",
            subtitle="Raw strings passed to renderer (may fail if not valid SVG).",
            n_rows=n_rows,
            pred_col="raw_pred",
            right_title="Raw output (rendered)",
        )

    def display_post_text(self, model_index: int, *, n_rows: int | None, preview_chars: int) -> None:
        p = self._get_payload(model_index)
        sample_df = p["sample_df"]
        method = p["postprocess_method"]
        display_text_comparisons(
            sample_df,
            title=f"Model {model_index} — text: target vs postprocessed",
            n_rows=n_rows,
            preview_chars=preview_chars,
            pred_col="pred_svg",
            right_heading=f"Postprocessed ({method})",
        )

    def display_post_rendered(self, model_index: int, *, n_rows: int | None) -> None:
        p = self._get_payload(model_index)
        sample_df = p["sample_df"]
        display_rendered_comparisons(
            sample_df,
            title=f"Model {model_index} — rendered: target vs postprocessed",
            n_rows=n_rows,
            pred_col="pred_svg",
            right_title="Postprocessed (rendered)",
        )

    def display_raw_metrics(self, model_index: int) -> None:
        p = self._get_payload(model_index)
        if not p.get("show_raw_metrics", True):
            display(HTML("<p><i>SHOW_RAW_METRICS is False; skipping.</i></p>"))
            return
        sample_df = p["sample_df"]
        raw_metrics_df = enrich_for_display(sample_df.copy(), "raw_pred")
        mid = p["meta"]["model_id"]
        display_prediction_summary(raw_metrics_df, heading=f"Quality metrics (raw decode) — {mid}")

    def display_postprocessed_metrics(self, model_index: int) -> None:
        p = self._get_payload(model_index)
        sample_df = p["sample_df"]
        mid = p["meta"]["model_id"]
        display_prediction_summary(
            sample_df,
            heading=f"Quality metrics (postprocessed) — {mid}",
        )

    def display_all_comparisons_for_model(
        self,
        model_index: int,
        *,
        n_text_rows: int | None,
        n_render_rows: int | None,
        preview_chars: int,
    ) -> None:
        """Raw + postprocessed text and rendered comparisons for one cached model."""
        p = self._get_payload(model_index)
        mid = p["meta"]["model_id"]
        method = p["postprocess_method"]
        display(
            HTML(
                f'<h3 style="margin-top:1em;">Model index {model_index} — <code>{html_module.escape(mid)}</code></h3>'
            )
        )
        display(HTML("<h4>Raw output vs target</h4>"))
        self.display_raw_metrics(model_index)
        self.display_raw_text(model_index, n_rows=n_text_rows, preview_chars=preview_chars)
        self.display_raw_rendered(model_index, n_rows=n_render_rows)
        display(
            HTML(
                f"<h4>Postprocessed vs target (<code>{html_module.escape(method)}</code>)</h4>"
            )
        )
        self.display_postprocessed_metrics(model_index)
        self.display_post_text(model_index, n_rows=n_text_rows, preview_chars=preview_chars)
        self.display_post_rendered(model_index, n_rows=n_render_rows)
