import gc
import logging
import random
import warnings

import numpy as np
import torch

_training_warnings_suppressed = False


def suppress_training_warnings():
    """Reduce noisy library warnings during training runs.

    Args:
        None.

    Returns:
        None: This function mutates global warning and logging state for the current Python
            process so notebook training output stays readable.

    Notes:
        The suppression is applied only once per process. It silences broad warning streams,
        lowers Hugging Face and datasets logging verbosity, and quiets the TRL SFT trainer
        logger while preserving tqdm progress bars.
    """
    global _training_warnings_suppressed
    if _training_warnings_suppressed:
        return
    _training_warnings_suppressed = True
    warnings.filterwarnings("ignore")
    from datasets.utils import logging as ds_logging
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    ds_logging.set_verbosity_error()
    # TRL SFTTrainer warns when tokenize(prompt) != prefix of tokenize(prompt+completion) (BPE
    # boundary / instruct formatting); training still runs. Silence that logger only, not all of TRL.
    logging.getLogger("trl.trainer.sft_trainer").setLevel(logging.ERROR)


def set_seed(seed=42):
    """Seed Python, NumPy, and Torch random number generators.

    Args:
        seed (int, optional): Integer seed applied to Python's ``random`` module, NumPy, and
            Torch. Defaults to ``42``.

    Returns:
        None: RNG state is updated in place for the current process and all visible CUDA devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory():
    """Run garbage collection and free cached CUDA memory when available.

    Args:
        None.

    Returns:
        None: Python objects may be collected, and Torch's CUDA allocator cache is emptied if a
            GPU is available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
