import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from .base_cls import ProbabilitySimplexMapping

class ScalableSoftmax(ProbabilitySimplexMapping):
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(beta_init).log()
        )

    @property
    def beta(self):
        return self.log_beta.exp()

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        n = logits.size(dim)
        scale = 1.0 + self.beta * math.log(n)
        return torch.softmax(scale * logits, dim=dim)
