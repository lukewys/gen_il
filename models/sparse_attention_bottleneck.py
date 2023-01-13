import math
import torch
from torch import nn, optim
from torch.nn import functional as F


class SparseAttnBottleneck(nn.Module):
    def __init__(self, voc_size, dim, num_topk, temperature=1.0, use_softmax=True, mirror_neg_code=False):
        super().__init__()
        self.voc_size = voc_size
        self.codebook = nn.Parameter(torch.randn(voc_size, dim))
        # initialize the codebook just like linear layer
        nn.init.kaiming_uniform_(self.codebook, a=math.sqrt(5))
        self.codebook.cuda()  # just a hack
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.num_topk = num_topk
        self.temperature = temperature
        self.use_softmax = use_softmax
        self.mirror_neg_code = mirror_neg_code

    def max_neg_value(self, tensor):
        return -torch.finfo(tensor.dtype).max

    def attn_map_to_dots(self, top_ind, top_value):
        batch = top_ind.shape[0]
        dots = torch.zeros([batch, self.codebook.shape[0]]).to(top_ind.device)
        dots[torch.arange(batch).unsqueeze(1), top_ind] = top_value
        return dots

    def forward(self, x):
        # x: (batch, dim)
        q = self.to_q(x)

        # here I first assume k and v are both comes from the codebook
        k = self.to_k(self.codebook)
        v = self.to_v(self.codebook)

        if self.mirror_neg_code:
            k = torch.cat([k, -k], dim=0)
            v = torch.cat([v, -v], dim=0)

        # attention
        dots = torch.einsum('bd,kd->bk', q, k)

        # top k of attention
        top_value, top_ind = dots.topk(self.num_topk, dim=-1)
        vk = top_value[..., -1].unsqueeze(-1).expand_as(dots)
        mask = dots < vk
        mask_value = self.max_neg_value(dots)
        dots.masked_fill_(mask, mask_value)

        if self.use_softmax:
            # softmax
            attn = F.softmax(dots / self.temperature, dim=-1)
        else:
            # In the case of not using softmax, we directly use the logits.
            # We also do not apply the temperature in this case.
            attn = dots

        # get output by applying attention weight to v
        out = torch.einsum('bk,kd->bd', attn, v)
        return out, top_value, top_ind, dots
