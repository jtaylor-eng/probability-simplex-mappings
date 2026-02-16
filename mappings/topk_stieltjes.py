import torch

from .base_cls import ProbabilitySimplexMapping


class TopKStieltjes(ProbabilitySimplexMapping):
    """Top-K attention with Stieltjes transform: mask to top-k then apply Stieltjes (like topk_attn with softmax)."""
    def __init__(
        self,
        k: int,
        q: float = 1.0,
        num_iter: int = 16,
        eps: float = 1e-9,
    ):
        super().__init__()
        assert k > 0
        self._k = k
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _line_search_bs(self, shifted_logits, dim, lb, ub):
        for _ in range(self._num_iter):
            mid = (lb + ub) / 2.0
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
                dim=dim,
                keepdim=True
            ) - 1
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)
        return lb, ub

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        topk_vals, topk_idx = logits.topk(self._k, dim=dim)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(dim, topk_idx, topk_vals)

        x_max = torch.max(masked_logits, dim=dim, keepdim=True).values
        x_i = masked_logits - x_max

        lb = torch.full_like(x_max, self._eps)
        ub = torch.full_like(x_max, masked_logits.shape[dim] ** (1.0 / self._q))

        lb, ub = self._line_search_bs(
            shifted_logits=x_i,
            dim=dim,
            lb=lb,
            ub=ub,
        )
        lambda_1 = (lb + ub) / 2.0
        return torch.pow((lambda_1 - x_i).clamp(min=self._eps), -self._q)
