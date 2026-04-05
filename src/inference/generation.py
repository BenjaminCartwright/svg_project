import torch

from src.inference.postprocess import sanitize_svg_prediction


def generate_svg_prediction(prompt_text, tokenizer, model, max_new_tokens=768):
    """Generate one sanitized SVG from a causal language model prompt.

    Args:
        prompt_text (str): Fully formatted prompt text expected by the causal LM.
        tokenizer: Tokenizer with ``pad_token_id`` and ``eos_token_id`` configured.
        model: Causal language model exposing ``device`` and ``generate()``.
        max_new_tokens (int, optional): Maximum number of generated continuation tokens.
            Defaults to ``768``.

    Returns:
        str: Sanitized SVG string derived from the generated continuation. If the decoded output
            echoes ``prompt_text``, the echoed prefix is stripped before sanitization.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text):].strip()
    return sanitize_svg_prediction(decoded)


@torch.no_grad()
def generate_svg_raw_prediction(prompt_text, tokenizer, model, max_new_tokens=768):
    """Generate assistant continuation only: no fragment extraction or ``clean_svg``.

    Use this for training-notebook smoke tests and for evaluation pipelines that apply
    post-processing separately.

    Args:
        prompt_text (str): Fully formatted prompt (e.g. chat-style instruction).
        tokenizer: Tokenizer with ``pad_token_id`` and ``eos_token_id`` configured.
        model: Causal language model exposing ``device`` and ``generate()``.
        max_new_tokens (int, optional): Maximum generated continuation tokens.

    Returns:
        str: Decoded continuation after stripping an echoed prompt prefix.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text) :].strip()
    return decoded
