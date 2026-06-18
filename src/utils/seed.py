import torch
import numpy as np
import random


def set_seed(seed=42):
    """Lock all random number generators so every run gives identical results.

    TODO:
    1. torch.manual_seed(seed)
    2. torch.cuda.manual_seed_all(seed)
    3. np.random.seed(seed)
    4. random.seed(seed)
    5. torch.backends.cudnn.deterministic = True
    6. torch.backends.cudnn.benchmark = False
       (benchmark=True is faster but picks non-deterministic algorithms)

    Call set_seed(42) as the very first line of 03_train_lora.py and 04_evaluate.py.

    Args:
        seed: integer seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
