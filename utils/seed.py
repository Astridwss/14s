"""随机种子工具 —— 纯函数，无状态。"""
import random
import numpy as np


def set_seeds(seed: int):
    """设置所有随机种子。"""
    np.random.seed(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
