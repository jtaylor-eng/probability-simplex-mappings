import torch
from entmax import entmax_bisect
from .base_cls import ProbabilitySimplexMapping

class AlphaEntmax(ProbabilitySimplexMapping):
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        assert alpha > 1.0
        self.alpha = alpha

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs
    ) -> torch.Tensor:
        return entmax_bisect(
            logits,
            alpha=self.alpha,
            dim=dim
        )
