from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_t5_model_and_tokenizer(model_name: str = "t5-small"):
    """Load a pretrained T5 model and tokenizer for seq2seq experiments.

    Args:
        model_name (str, optional): Hugging Face model identifier or local path. Defaults to
            ``"t5-small"``.

    Returns:
        tuple[transformers.T5ForConditionalGeneration, transformers.T5Tokenizer]: ``(model,
        tokenizer)`` loaded from the requested checkpoint.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer
