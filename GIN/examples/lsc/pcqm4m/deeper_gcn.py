import torch
import torch.nn as nn


class DeeperGCN(nn.Module):
    def __init__(self, num_tasks=1, num_layers=3, *, args=None):
        super().__init__()