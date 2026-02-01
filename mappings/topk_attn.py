import torch
from .base_cls import ProbabilitySimplexMapping


class TopKAttention(ProbabilitySimplexMapping):
    def __init__(self, k: int):
        super().__init__()
        assert k > 0
        self.k = k

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        topk_vals, topk_idx = logits.topk(self.k, dim=dim)

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(dim, topk_idx, topk_vals)

        return torch.softmax(masked_logits, dim=dim)
