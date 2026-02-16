# import torch

# from .base_cls import ProbabilitySimplexMapping


# class ExponentialStieltjesTransform(ProbabilitySimplexMapping):
#     """\"Exponential Stieltjes\" mapping.

#     Uses the same Stieltjes-style binary search to obtain a scale λ such that
#     \\sum_i (λ - x_i)^(-q) = 1, then exponentiates these Stieltjes scores and
#     renormalizes with a softmax:

#         t_i = (λ - x_i)^(-q)
#         p_i ∝ exp(t_i)

#     This keeps the Stieltjes root-finding structure but adds an exponential
#     nonlinearity on top of the Stieltjes scores.
#     """

#     def __init__(
#         self,
#         q: float = 1.0,
#         num_iter: int = 16,
#         eps: float = 1e-9,
#     ):
#         super().__init__()
#         self._q = q
#         self._num_iter = num_iter
#         self._eps = eps

#     def _line_search_bs(self, shifted_logits, dim, lb, ub):
#         for _ in range(self._num_iter):
#             mid = (lb + ub) / 2.0

#             prob_sum = torch.sum(
#                 torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
#                 dim=dim,
#                 keepdim=True,
#             ) - 1

#             lb = torch.where(prob_sum > 0, mid, lb)
#             ub = torch.where(prob_sum <= 0, mid, ub)

#         return lb, ub

#     def translate_logits(
#         self,
#         logits: torch.Tensor,
#         dim: int,
#         **kwargs,
#     ) -> torch.Tensor:
#         # Center for numerical stability.
#         x_max = torch.max(logits, dim=dim, keepdim=True).values
#         x_i = logits - x_max

#         lb = torch.full_like(x_max, self._eps)
#         ub = torch.full_like(x_max, logits.shape[dim] ** (1.0 / self._q))

#         lb, ub = self._line_search_bs(
#             shifted_logits=x_i,
#             dim=dim,
#             lb=lb,
#             ub=ub,
#         )
#         lambda_1 = (lb + ub) / 2.0

#         # Stieltjes scores then exponential normalization.
#         t = torch.pow((lambda_1 - x_i).clamp(min=self._eps), -self._q)
#         return torch.softmax(t, dim=dim)

