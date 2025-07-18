import torch.nn as nn
from .fractal import GlobalShared
from .heads   import PersonalHead


class CombinedModel(nn.Module):
    def __init__(self, n_cls, w, depth, seed,
                 sample_loader=None):
        super().__init__()
        self.shared = GlobalShared(sample_loader=sample_loader)
        self.head   = PersonalHead(n_cls, w, depth, seed)
    def forward(self, x):
        return self.head(self.shared(x))