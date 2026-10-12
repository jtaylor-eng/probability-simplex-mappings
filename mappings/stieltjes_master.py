"""
stieltjes_master.py — Reference implementations of all Stieltjes-based simplex mappings.

Mathematical background (from proposal):
  For logits x ∈ ℝⁿ, the general Tsallis-q regularised argmax solves:
      max_{y ∈ Δₙ}  xᵀy + C · R_q(y)
  where:
      R_q(y) = -q/(q-1) · Σᵢ yᵢ^(1-1/q)   (Tsallis entropy, q > 1)
      R_1(y) = -Σᵢ log yᵢ                   (Burg/log-barrier entropy, limiting q→1 case)

  The optimal solution is the Stieltjes distribution:
      yᵢ* = (λ_q - xᵢ)^(-q)
  where λ_q > max xᵢ is the unique root of:
      f(λ) = Σᵢ (λ - xᵢ)^(-q) = 1

  f is strictly decreasing in λ, so bisection is guaranteed to converge.

  BOUNDS (after shifting xᵢ ← xᵢ - max xᵢ so that x_max = 0):
      lb = ε             (any λ > x_max = 0 is a valid lower bound)
      ub = n^(1/q)       (tight upper bound: f(n^(1/q)) ≤ 1 always,
                          with equality iff all xᵢ are equal)
  Proof of upper bound:
      After shifting, all xᵢ ≤ 0.  For any xᵢ ≤ 0:
          (n^(1/q) - xᵢ)^(-q) ≤ n^(-1/q · q) = 1/n
      Therefore f(n^(1/q)) = Σᵢ (n^(1/q) - xᵢ)^(-q) ≤ n · (1/n) = 1.  □

  KEY IMPLEMENTATION NOTES:
  - The upper bound must be DYNAMIC (n^(1/q) with runtime n) to handle
    sequences of any length, including long OOD test sequences.
    Using a static bound (e.g. 4096^(1/q)) breaks for n > 4096.
  - Explicit normalisation is added as a defensive measure against
    floating-point drift accumulated over 15 bisection steps.

References:
  Base Stieltjes:              user proposal (Tsallis/Burg regularisation)
  ScalableStieltjes:           arxiv 2501.19399 (scalable softmax/entmax)
  AdaptiveTemperatureStieltjes: arxiv 2410.01104 (entropy-adaptive temperature)
  AdaptiveScalableStieltjes:   arxiv 2506.16640 (ASEntmax, adapted for Stieltjes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Base Stieltjes Transform
# ---------------------------------------------------------------------------

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
        x_max = logits.max(dim=dim, keepdim=True).values
        shifted = logits - x_max

        lam = _bisect_lambda(shifted, dim, self.q, self.num_iter, self.eps)
        return _stieltjes_from_lambda(shifted, lam, dim, self.q, self.eps)


# ---------------------------------------------------------------------------
# 2. Scalable Stieltjes
# ---------------------------------------------------------------------------

class ScalableStieltjes(nn.Module):
    """
    Length-scalable Stieltjes mapping.  Multiplies logits by a learnable
    log-linear function of the sequence length before applying the base
    Stieltjes transform:

        scale(n) = 1 + β · log(n),    β = exp(log_β)   (always positive)
        y = Stieltjes(scale(n) · x,  q)

    Motivation (arxiv 2501.19399): softmax/entmax attention distributions
    become increasingly uniform as sequence length grows.  Scaling logits
    by log(n) compensates for this, giving better length extrapolation.

    The same principle applies to Stieltjes: scaling sharpens the
    distribution at longer sequences, counteracting the dilution effect.

    Args:
        q:         Tsallis moment order
        num_iter:  bisection iterations
        eps:       numerical floor
        beta_init: initial value of β (stored as log_β)
    """

    def __init__(self, q: float = 16.0, num_iter: int = 15, eps: float = 1e-9,
                 beta_init: float = 1.0):
        super().__init__()
        self.q = q
        self.num_iter = num_iter
        self.eps = eps
        # Store log β so that β = exp(log_β) > 0 always
        self._log_beta = nn.Parameter(torch.tensor(math.log(max(beta_init, eps))))

    @property
    def beta(self) -> torch.Tensor:
        return self._log_beta.exp()

    def translate_logits(self, logits: torch.Tensor, dim: int = -1,
                         **kwargs) -> torch.Tensor:
        n = logits.size(dim)
        # scale is a scalar (no gradient w.r.t. n, but gradient w.r.t. β flows)
        scale = 1.0 + self.beta * math.log(max(n, 2))
        scaled = scale * logits

        x_max = scaled.max(dim=dim, keepdim=True).values
        shifted = scaled - x_max

        lam = _bisect_lambda(shifted, dim, self.q, self.num_iter, self.eps)
        return _stieltjes_from_lambda(shifted, lam, dim, self.q, self.eps)


# ---------------------------------------------------------------------------
# 3. Adaptive Temperature Stieltjes
# ---------------------------------------------------------------------------

class AdaptiveTemperatureStieltjes(nn.Module):
    """
    Entropy-adaptive temperature Stieltjes (arxiv 2410.01104, adapted).

    The original paper (for softmax) observes that attention entropy H
    is a reliable proxy for over-diffuseness.  A polynomial T(H) is fitted
    to produce a sharpening temperature β ≥ 1, then applied as:

        y = Stieltjes(β(H) · x,  q)

    where β(H) is computed from the Shannon entropy of the *softmax*
    distribution (not the Stieltjes distribution), following the original
    paper's schedule.  This decouples the expensive Stieltjes computation
    from the entropy estimate.

    Polynomial coefficients (degree-4 polynomial in H, Horner form):
        T(H) = -0.037 H⁴ + 0.481 H³ - 2.3 H² + 4.917 H - 1.791
    fitted so T(H) ≈ 1 near H=0 (already sharp) and T(H) > 1 for H > 0.5
    (diffuse → sharpen).  Temperature is clamped to [1, 10].

    Note: The polynomial was originally fitted against *softmax* behaviour.
    Its transfer to Stieltjes is a reasonable engineering approximation;
    deeper calibration would require fitting against Stieltjes entropy.

    Args:
        q:      Tsallis moment order
        coeffs: degree-4 polynomial coefficients (highest power first)
    """

    _DEFAULT_COEFFS = [-0.037, 0.481, -2.3, 4.917, -1.791]

    def __init__(self, q: float = 16.0, num_iter: int = 15, eps: float = 1e-9,
                 coeffs=None):
        super().__init__()
        self.q = q
        self.num_iter = num_iter
        self.eps = eps
        if coeffs is None:
            coeffs = self._DEFAULT_COEFFS
        self.register_buffer("poly_coeffs", torch.tensor(coeffs, dtype=torch.float32))

    @staticmethod
    def _polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Horner's method evaluation of polynomial."""
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs:
            out = out * x + c
        return out

    def translate_logits(self, logits: torch.Tensor, dim: int = -1,
                         **kwargs) -> torch.Tensor:
        # --- Compute softmax entropy (no gradient; used as a proxy only) ---
        with torch.no_grad():
            logits_f32 = logits.float()
            probs = F.softmax(logits_f32, dim=dim)
            log_probs = F.log_softmax(logits_f32, dim=dim)
            # keepdim=True preserves broadcastability with logits
            H = -(probs * log_probs).sum(dim=dim, keepdim=True)

        # --- Temperature from polynomial ---
        T = self._polyval(self.poly_coeffs.to(logits.device), H.float())
        T = T.clamp(min=1.0, max=10.0)
        # Only sharpen (β ≥ 1); leave sharp distributions unchanged
        T = torch.where(H > 0.5, T, torch.ones_like(T))
        T = T.to(logits.dtype)

        # --- Apply temperature and Stieltjes ---
        scaled = logits * T
        x_max = scaled.max(dim=dim, keepdim=True).values
        shifted = scaled - x_max

        lam = _bisect_lambda(shifted, dim, self.q, self.num_iter, self.eps)
        return _stieltjes_from_lambda(shifted, lam, dim, self.q, self.eps)


# ---------------------------------------------------------------------------
# 4. Adaptive Scalable Stieltjes
# ---------------------------------------------------------------------------

class AdaptiveScalableStieltjes(nn.Module):
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
