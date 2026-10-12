import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .base_cls import ProbabilitySimplexMapping


def _bisect_lambda(shifted_logits: torch.Tensor, dim: int, q: float,
                   num_iter: int, eps: float) -> torch.Tensor:
    n = shifted_logits.size(dim)
    ub = torch.full_like(
        shifted_logits.max(dim=dim, keepdim=True).values,
        n ** (1.0 / q),
    )
    lb = torch.full_like(ub, eps)
    for _ in range(num_iter):
        mid = (lb + ub) * 0.5
        f_mid = torch.sum(
            torch.pow((mid - shifted_logits).clamp(min=eps), -q),
            dim=dim, keepdim=True,
        ) - 1.0
        lb = torch.where(f_mid > 0.0, mid, lb)
        ub = torch.where(f_mid <= 0.0, mid, ub)
    return (lb + ub) * 0.5


def _stieltjes_from_lambda(shifted_logits: torch.Tensor, lam: torch.Tensor,
                           dim: int, q: float, eps: float) -> torch.Tensor:
    probs = torch.pow((lam - shifted_logits).clamp(min=eps), -q)
    return probs / probs.sum(dim=dim, keepdim=True).clamp(min=eps)


class AdaptiveScalableStieltjes(ProbabilitySimplexMapping):
    """
    Mirrors AdaptiveScalableEntmax (as_entmax.py) but with Stieltjes in place
    of alpha-entmax.  β and γ are both learnable parameters.

    Scaling:
        scale(q, K) = δ + β(q) · (log K)^γ
        y = Stieltjes(scale · x, q_order)

    β init: w_β = 0  →  β = softplus(0) ≈ 0.693; at K=16 scale ≈ 2.9,
    giving a strong length-scaling inductive bias from step 1.
    γ init: exp(0) = 1.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 1,
                 gamma: float = 1.0, delta: float = 1.0,
                 q_order: float = 16.0, num_iter: int = 15, eps: float = 1e-9):
        super().__init__()
        self.delta = delta
        self.q_order = q_order
        self.num_iter = num_iter
        self.eps = eps

        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
        self._log_gamma = nn.Parameter(torch.tensor(math.log(max(gamma, 1e-6))))

    def translate_logits(self, logits: torch.Tensor, dim: int = -1,
                         queries: torch.Tensor | None = None,
                         **kwargs) -> torch.Tensor:
        if queries is None:
            queries = kwargs.get("queries", None)
        if queries is None:
            raise ValueError("AdaptiveScalableStieltjes requires queries.")

        added_head_dim = False
        if queries.dim() == 3:
            queries = queries.unsqueeze(1)
            added_head_dim = True
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            added_head_dim = True

        K = logits.size(dim if dim >= 0 else logits.dim() + dim)
        beta = F.softplus(torch.einsum("bhqd,hd->bhq", queries, self.w_beta))  # (B, H, Q)
        scale = self.delta + beta * (math.log(max(float(K), 2.0)) ** self._log_gamma.exp())
        scale = scale.unsqueeze(-1)  # (B, H, Q, 1)

        scaled_logits = scale * logits

        x_max = scaled_logits.max(dim=dim, keepdim=True).values
        shifted = scaled_logits - x_max
        lam = _bisect_lambda(shifted, dim, self.q_order, self.num_iter, self.eps)
        out = _stieltjes_from_lambda(shifted, lam, dim, self.q_order, self.eps)

        if added_head_dim and out.size(1) == 1:
            out = out.squeeze(1)
        return out


# class AdaptiveScalableStieltjes(ProbabilitySimplexMapping):
#     """
#     Query-dependent length-scalable Stieltjes mapping (mirrors ASEntmax,
#     arxiv 2506.16640, but with Stieltjes in place of alpha-entmax).
#
#     Scaling:
#         scale(q, K) = δ + β(q) · (log K)^γ
#         y = Stieltjes(scale · x,  q_order)
#
#     where β(q) is a per-head, per-query scalar computed from the query vector:
#         β_logit(q) = w_β · q + b_β
#         β(q)       = softplus(β_logit(q))    [always positive]
#
#     γ > 0 is either fixed or learnable (stored as log γ).
#     δ ≥ 0 is a fixed offset (default 1.0) ensuring scale ≥ δ.
#
#     Initialisation:
#         w_β = 0,  b_β = -5   →  β(q) = softplus(-5) ≈ 0.007
#         At init the scale ≈ 1 + 0.007 · log K ≈ 1, so training starts
#         near the unscaled Stieltjes distribution.
#
#     -∞ mask handling:
#         Causal attention masks set future-position logits to -∞.  Multiplying
#         -∞ by a positive scale returns -∞ (fine), but 0 · (-∞) = NaN.  We
#         therefore zero-out masked positions before scaling and restore -∞
#         afterwards.
#
#     Args:
#         d_model:   head dimension (size of query vectors)
#         n_heads:   number of attention heads
#         gamma:     exponent on log K (None → learnable, initialised to 1)
#         delta:     additive offset (default 1.0)
#         q_order:   Tsallis moment order
#         num_iter:  bisection iterations
#         eps:       numerical floor
#     """
#
#     def __init__(self, d_model: int = 64, n_heads: int = 1,
#                  gamma: float | None = None, delta: float = 1.0,
#                  q_order: float = 16.0, num_iter: int = 15, eps: float = 1e-9):
#         super().__init__()
#         self.delta = delta
#         self.q_order = q_order
#         self.num_iter = num_iter
#         self.eps = eps
#
#         # Query projection for β: initialise weights to 0, bias to -5
#         self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
#         self.b_beta = nn.Parameter(torch.full((n_heads,), -5.0))
#
#         # γ: learnable (exp-parameterised) or fixed
#         if gamma is None:
#             self._log_gamma = nn.Parameter(torch.tensor(0.0))   # γ = exp(0) = 1
#             self._gamma_fixed = False
#         else:
#             self.register_buffer("_log_gamma",
#                                  torch.tensor(math.log(max(gamma, 1e-6))))
#             self._gamma_fixed = True
#
#         self._base = StieltjesTransform(q=q_order, num_iter=num_iter, eps=eps)
#
#     @property
#     def gamma(self) -> torch.Tensor:
#         return self._log_gamma.exp()
#
#     def translate_logits(self, logits: torch.Tensor, dim: int = -1,
#                          queries: torch.Tensor | None = None,
#                          **kwargs) -> torch.Tensor:
#         if queries is None:
#             raise ValueError("AdaptiveScalableStieltjes requires queries.")
#
#         # Ensure 4-D (B, H, Q, K) / (B, H, Q, D) for both tensors
#         added_head_dim = False
#         if queries.dim() == 3:      # (B, Q, D) → (B, 1, Q, D)
#             queries = queries.unsqueeze(1)
#             added_head_dim = True
#         if logits.dim() == 3:       # (B, Q, K) → (B, 1, Q, K)
#             logits = logits.unsqueeze(1)
#             added_head_dim = True
#
#         K = logits.size(dim if dim >= 0 else logits.dim() + dim)
#         # Use max(K, 2) so that log(K) ≥ log(2) > 0 even during single-token decode
#         log_K = math.log(max(float(K), 2.0))
#
#         # β(q): (B, H, Q)
#         beta_logit = (torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
#                       + self.b_beta.view(1, -1, 1))
#         beta = F.softplus(beta_logit)           # (B, H, Q), always > 0
#
#         # scale: (B, H, Q, 1)
#         scale = self.delta + beta * (log_K ** self.gamma)
#         scale = scale.unsqueeze(-1)
#
#         # -∞ mask handling: 0 · (-∞) = NaN → temporarily zero out, restore after
#         is_masked = logits.isinf() & (logits < 0)
#         safe_logits = logits.masked_fill(is_masked, 0.0)
#         scaled = scale * safe_logits
#         scaled = scaled.masked_fill(is_masked, float('-inf'))
#
#         # Bisection (x_max shift done inside StieltjesTransform)
#         out = self._base.translate_logits(scaled, dim=dim)
#
#         if added_head_dim and out.size(1) == 1:
#             out = out.squeeze(1)
#         return out