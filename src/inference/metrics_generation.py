"""Generation-scoring metrics that are not the same as SFT ``eval_loss``.

``eval_loss`` from supervised fine-tuning is teacher-forced cross-entropy on the **gold** target
tokens over the training batch. It does **not** change when you only change ``max_new_tokens`` at
inference.

The functions here compute **causal LM cross-entropy on a chosen continuation** (tokenized
immediately after the prompt), with loss masked to **continuation tokens only**. Two uses:

* **continuation_nll_mean** on ``raw_pred``: model's NLL for the **greedy decoded** continuation;
  this generally **does** depend on decoding budget because the string changes.
* Same function with ``target_svg``: **gold teacher NLL** (probability assigned to the reference
  SVG under the model); independent of ``max_new_tokens`` when you only vary decoding.
"""

from __future__ import annotations

import math

import torch


@torch.inference_mode()
def mean_nll_continuation_only(
    model,
    tokenizer,
    prompt_text: str,
    continuation_text: str,
) -> float:
    """Mean cross-entropy over continuation token positions (prompt positions masked out).

    Tokenization matches ``generate_svg_raw_prediction``: ``prompt_text`` uses the tokenizer's
    default ``add_special_tokens`` (same as ``tokenizer(..., return_tensors="pt")``); continuation
    is encoded with ``add_special_tokens=False`` and concatenated to the prompt token ids.

    Args:
        model: Causal LM (possibly PEFT-wrapped) in eval mode.
        tokenizer: Paired tokenizer.
        prompt_text: Full formatted prompt up to the assistant turn (no answer).
        continuation_text: Decoded assistant continuation (generated or gold SVG text).

    Returns:
        Scalar mean NLL (natural log) over labeled continuation positions, or NaN if the
        continuation encodes to zero tokens.
    """
    continuation_text = "" if continuation_text is None else str(continuation_text)
    if not continuation_text.strip():
        return float("nan")

    device = model.device
    enc_p = tokenizer(prompt_text, return_tensors="pt")
    enc_c = tokenizer(continuation_text, add_special_tokens=False, return_tensors="pt")
    if enc_c["input_ids"].shape[1] == 0:
        return float("nan")

    input_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    pl = int(enc_p["input_ids"].shape[1])
    labels[:, :pl] = -100

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if out.loss is None:
        return float("nan")
    return float(out.loss)


def mean_nll_over_series(
    model,
    tokenizer,
    prompts: list[str],
    continuations: list[str],
    *,
    show_progress: bool = False,
    progress_desc: str = "NLL",
) -> float:
    """Batch-mean of per-example ``mean_nll_continuation_only`` (skips NaN rows)."""
    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have the same length")
    if not prompts:
        return float("nan")

    iterator = range(len(prompts))
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=progress_desc)
        except Exception:
            pass

    vals: list[float] = []
    for i in iterator:
        v = mean_nll_continuation_only(model, tokenizer, prompts[i], continuations[i])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))
