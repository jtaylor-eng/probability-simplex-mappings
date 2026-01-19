import torch
import torch.nn.functional as F
from entmax import entmax15
from .base_cls import ProbabilitySimplexMapping

class AlphaEntmax(ProbabilitySimplexMapping):
    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
        return entmax15(logits, dim=dim)