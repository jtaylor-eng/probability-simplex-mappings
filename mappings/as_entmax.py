import torch
import torch.nn as nn
import math
from entmax import entmax15
from .base_cls import ProbabilitySimplexMapping

class AdaptiveScalableEntmax(ProbabilitySimplexMapping):
    def __init__(self, d_model, n_heads, gamma=1.0, delta=1.0):
        super().__init__()
        self.delta = delta

        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
        self.w_gamma = gamma

    def translate_logits(self, logits, queries, dim=-1):
        K = logits.shape[-1]

        beta = torch.nn.functional.softplus(
            torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
        )

        scale = self.delta + beta * (math.log(K) ** self.w_gamma)
        scale = scale.unsqueeze(-1)

        scaled_logits = scale * logits
        return entmax15(scaled_logits, dim=dim)
