import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # 通用复现附加项
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64
    except Exception:
        pass

    print(f"[Seed] Global seed set to {seed}, deterministic={deterministic}")
