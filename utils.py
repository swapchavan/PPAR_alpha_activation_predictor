# utils.py
import random
import numpy as np
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
