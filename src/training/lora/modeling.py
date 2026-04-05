import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def make_quant_config():
    """Build the 4-bit quantization configuration used for LoRA training.

    Args:
        None.

    Returns:
        transformers.BitsAndBytesConfig: Bitsandbytes configuration enabling 4-bit NF4
            quantization with double quantization and a CUDA-aware compute dtype.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer_and_base_model(model_id):
    """Load a tokenizer and quantized causal language model.

    Args:
        model_id (str): Hugging Face model identifier or local path for the base model.

    Returns:
        tuple[Any, Any]: ``(tokenizer, model)`` where the tokenizer has a pad token configured and
            the model has been prepared for k-bit training.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=make_quant_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def apply_lora(model, config):
    """Attach a LoRA adapter to a causal language model.

    Args:
        model: Base model returned by ``load_tokenizer_and_base_model``.
        config (Mapping[str, Any]): Hyperparameter mapping containing ``lora_r``,
            ``lora_alpha``, and ``lora_dropout``.

    Returns:
        peft.PeftModel: Model wrapped with trainable LoRA adapters on the configured projection
            modules.
    """
    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def load_adapter_for_inference(adapter_dir, model_id):
    """Load a saved adapter on top of its base model for inference.

    Args:
        adapter_dir (str | pathlib.Path): Directory containing saved PEFT adapter weights.
        model_id (str): Base model identifier compatible with the adapter.

    Returns:
        tuple[Any, peft.PeftModel]: ``(tokenizer, model)`` ready for evaluation and generation.
    """
    tokenizer, base_model = load_tokenizer_and_base_model(model_id)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return tokenizer, model


def load_inference_adapter(adapter_dir, model_id):
    """Alias for loading tokenizer and PEFT adapter for inference notebooks.

    Args:
        adapter_dir (str | pathlib.Path): Directory containing saved adapter weights.
        model_id (str): Base model identifier compatible with the adapter.

    Returns:
        tuple[Any, peft.PeftModel]: Same ``(tokenizer, model)`` pair returned by
            ``load_adapter_for_inference``.
    """
    return load_adapter_for_inference(adapter_dir, model_id)
