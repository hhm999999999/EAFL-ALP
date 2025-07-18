import torch, torch.nn as nn, math, numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def subtree_info(block, sample_loader, n_batch=3):
    outs = []
    device = next(block.parameters()).device
    for i, (x, _) in enumerate(sample_loader):
        if i == n_batch:
            break
        outs.append(block(x.to(device)).cpu())
    if not outs:
        return 0.0
    z = torch.cat(outs, 0)
    return float(z.var(dim=0).mean())


class FractalBlock(nn.Module):

    def __init__(self, d_in, d_out, depth,
                 w=1.0, seed=0,
                 sample_loader: DataLoader | None = None,
                 thresh: float = 1e-4):
        super().__init__()
        self.depth = depth
        self.pruned = False


        if depth == 0:
            self.lin = nn.Linear(d_in, d_out, bias=False)
            nn.init.kaiming_uniform_(self.lin.weight, a=math.sqrt(5))
            self.ln = nn.LayerNorm(d_out)

            rng = np.random.RandomState(seed)
            keep = max(1, int(round(d_out * w)))
            mask = np.zeros(d_out, np.float32)
            mask[rng.choice(d_out, keep, replace=False)] = 1.
            self.register_buffer("mask", torch.from_numpy(mask))
            return


        self.left = FractalBlock(d_in, d_out, depth - 1,
                                 w, seed, sample_loader, thresh)
        self.right = FractalBlock(d_in, d_out, depth - 1,
                                  w, seed + 777, sample_loader, thresh)
        self.merge = nn.Linear(2 * d_out, d_out, bias=False)
        self.act = nn.ReLU()


        if sample_loader is not None:
            info_l = subtree_info(self.left,  sample_loader)
            info_r = subtree_info(self.right, sample_loader)
            if (info_l < thresh) and (info_r < thresh):

                del self.left, self.right, self.merge, self.act
                self.pruned = True

    def forward(self, x):
        if self.pruned:
            return x.new_zeros(x.size(0), self.out_dim)
        if self.depth == 0:
            return self.ln(torch.relu(self.lin(x))) * self.mask
        z = torch.cat([self.left(x), self.right(x)], dim=-1)
        return self.act(self.merge(z))


    @property
    def out_dim(self):
        return next(p for p in self.parameters()).shape[0]


class GlobalShared(nn.Module):
    def __init__(self, in_dim=7, feat=32, depth=2,
                 sample_loader: DataLoader | None = None):
        super().__init__()
        self.frac = FractalBlock(in_dim, feat, depth,
                                 sample_loader=sample_loader)

    def forward(self, x):
        return self.frac(x)
