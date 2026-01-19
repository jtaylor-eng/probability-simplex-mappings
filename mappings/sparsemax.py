import torch
import torch.nn.functional as F
from entmax import sparsemax

from .base_cls import ProbabilitySimplexMapping

class Sparsemax(ProbabilitySimplexMapping):
    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
        return sparsemax(logits, dim=dim)