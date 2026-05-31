import torch
import torch.nn as nn
import math
from entmax import entmax15
from .base_cls import ProbabilitySimplexMapping
import torch.nn.functional as F


def _bisect_lambda(shifted_logits: torch.Tensor, dim: int, q: float,
                   num_iter: int, eps: float) -> torch.Tensor:
    """
    Find λ > 0 such that Σᵢ (λ - shifted_xᵢ)^(-q) = 1,
    where shifted_xᵢ = xᵢ - max xᵢ ≤ 0.

    Returns λ as a tensor with the same shape as shifted_logits but with
    size 1 along `dim` (keepdim=True).
    """
    n = shifted_logits.size(dim)

    # Tight upper bound: f(n^(1/q)) ≤ 1 always (see module docstring).
    ub = torch.full_like(
        shifted_logits.max(dim=dim, keepdim=True).values,  # shape anchor
        n ** (1.0 / q),
    )
    lb = torch.full_like(ub, eps)

    for _ in range(num_iter):
        mid = (lb + ub) * 0.5
        # f(mid) - 1:  positive → mid too small → raise lb
        #              non-positive → mid is valid upper bound → lower ub
        f_mid = torch.sum(
            torch.pow((mid - shifted_logits).clamp(min=eps), -q),
            dim=dim, keepdim=True,
        ) - 1.0
        lb = torch.where(f_mid > 0.0, mid, lb)
        ub = torch.where(f_mid <= 0.0, mid, ub)

    return (lb + ub) * 0.5


def _stieltjes_from_lambda(shifted_logits: torch.Tensor, lam: torch.Tensor,
                           dim: int, q: float, eps: float) -> torch.Tensor:
    """
    Compute (λ - xᵢ)^(-q) and explicitly normalise to sum = 1.
    Explicit normalisation is a defensive measure against bisection drift.
    """
    probs = torch.pow((lam - shifted_logits).clamp(min=eps), -q)
    return probs / probs.sum(dim=dim, keepdim=True).clamp(min=eps)


class StieltjesTransform(nn.Module):
    """
    Base Stieltjes / Tsallis-q simplex mapping.

        yᵢ = (λ_q - xᵢ)^(-q),   Σᵢ yᵢ = 1

    q controls sparsity: larger q → sharper distribution.
    At q=1 this recovers the Burg-entropy (log-barrier) argmax.

    Args:
        q:        Tsallis moment order (float, > 0; typical: 16, 32, 64)
        num_iter: bisection iterations (15 gives ~1e-5 relative precision)
        eps:      numerical floor to prevent division by zero
    """

    def __init__(self, q: float = 16.0, num_iter: int = 15, eps: float = 1e-9):
        super().__init__()
        self.q = q
        self.num_iter = num_iter
        self.eps = eps

    def translate_logits(self, logits: torch.Tensor, dim: int = -1,
                         **kwargs) -> torch.Tensor:
        # Shift so x_max = 0 (numerical stability; does not change the distribution)
        # logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = logits.max(dim=dim, keepdim=True).values
        shifted = logits - x_max

        lam = _bisect_lambda(shifted, dim, self.q, self.num_iter, self.eps)
        return _stieltjes_from_lambda(shifted, lam, dim, self.q, self.eps)


class AdaptiveScalableStieltjes(ProbabilitySimplexMapping):
    """
    Query-dependent length-scalable Stieltjes mapping (mirrors ASEntmax,
    arxiv 2506.16640, but with Stieltjes in place of alpha-entmax).

    Scaling:
        scale(q, K) = δ + β(q) · (log K)^γ
        y = Stieltjes(scale · x,  q_order)

    where β(q) is a per-head, per-query scalar computed from the query vector:
        β_logit(q) = w_β · q + b_β
        β(q)       = softplus(β_logit(q))    [always positive]

    γ > 0 is either fixed or learnable (stored as log γ).
    δ ≥ 0 is a fixed offset (default 1.0) ensuring scale ≥ δ.

    Initialisation:
        w_β = 0,  b_β = -5   →  β(q) = softplus(-5) ≈ 0.007
        At init the scale ≈ 1 + 0.007 · log K ≈ 1, so training starts
        near the unscaled Stieltjes distribution.

    -∞ mask handling:
        Causal attention masks set future-position logits to -∞.  Multiplying
        -∞ by a positive scale returns -∞ (fine), but 0 · (-∞) = NaN.  We
        therefore zero-out masked positions before scaling and restore -∞
        afterwards.

    Args:
        d_model:   head dimension (size of query vectors)
        n_heads:   number of attention heads
        gamma:     exponent on log K (None → learnable, initialised to 1)
        delta:     additive offset (default 1.0)
        q_order:   Tsallis moment order
        num_iter:  bisection iterations
        eps:       numerical floor
    """

    def __init__(self, d_model: int = 64, n_heads: int = 1,
                 gamma: float | None = None, delta: float = 1.0,
                 q_order: float = 16.0, num_iter: int = 15, eps: float = 1e-9):
        super().__init__()
        self.delta = delta
        self.q_order = q_order
        self.num_iter = num_iter
        self.eps = eps

        # Query projection for β: initialise weights to 0, bias to -5
        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
        self.b_beta = nn.Parameter(torch.full((n_heads,), -5.0))

        # γ: learnable (exp-parameterised) or fixed
        if gamma is None:
            self._log_gamma = nn.Parameter(torch.tensor(0.0))   # γ = exp(0) = 1
            self._gamma_fixed = False
        else:
            self.register_buffer("_log_gamma",
                                 torch.tensor(math.log(max(gamma, 1e-6))))
            self._gamma_fixed = True

        self._base = StieltjesTransform(q=q_order, num_iter=num_iter, eps=eps)

    @property
    def gamma(self) -> torch.Tensor:
        return self._log_gamma.exp()

    def translate_logits(self, logits: torch.Tensor, dim: int = -1,
                         queries: torch.Tensor | None = None,
                         **kwargs) -> torch.Tensor:
        if queries is None:
            raise ValueError("AdaptiveScalableStieltjes requires queries.")

        # Ensure 4-D (B, H, Q, K) / (B, H, Q, D) for both tensors
        added_head_dim = False
        if queries.dim() == 3:      # (B, Q, D) → (B, 1, Q, D)
            queries = queries.unsqueeze(1)
            added_head_dim = True
        if logits.dim() == 3:       # (B, Q, K) → (B, 1, Q, K)
            logits = logits.unsqueeze(1)
            added_head_dim = True

        K = logits.size(dim if dim >= 0 else logits.dim() + dim)
        # Use max(K, 2) so that log(K) ≥ log(2) > 0 even during single-token decode
        log_K = math.log(max(float(K), 2.0))

        # β(q): (B, H, Q)
        beta_logit = (torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
                      + self.b_beta.view(1, -1, 1))
        beta = F.softplus(beta_logit)           # (B, H, Q), always > 0

        # scale: (B, H, Q, 1)
        scale = self.delta + beta * (log_K ** self.gamma)
        scale = scale.unsqueeze(-1)

        # -∞ mask handling: 0 · (-∞) = NaN → temporarily zero out, restore after
        is_masked = logits.isinf() & (logits < 0)
        safe_logits = logits.masked_fill(is_masked, 0.0)
        scaled = scale * safe_logits
        scaled = scaled.masked_fill(is_masked, float('-inf'))

        # Bisection (x_max shift done inside StieltjesTransform)
        out = self._base.translate_logits(scaled, dim=dim)

        if added_head_dim and out.size(1) == 1:
            out = out.squeeze(1)
        return out
