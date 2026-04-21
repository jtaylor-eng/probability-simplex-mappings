import torch
import torch.nn as nn
import math

from .base_cls import ProbabilitySimplexMapping
from .stieltjes import StieltjesTransform

class AdaptiveScalableStieltjes(ProbabilitySimplexMapping):
    """Adaptive Scalable Stieltjes: like ASEntmax but with Stieltjes transform.
    scale = delta + beta * (log K)^gamma, then Stieltjes(scale * logits) with given q.
    """
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 1,
        gamma: float | None = None,
        delta: float = 1.0,
        q: float = 32.0,
        num_iter: int = 15, 
        eps: float = 1e-9,
    ):
        super().__init__()
        self.delta = delta
        
        # 1. Initialize weights to exactly zero
        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
        # 2. Add a dedicated bias parameter initialized to -5.0
        self.b_beta = nn.Parameter(torch.full((n_heads,), -5.0))
        
        if gamma is None:
            self._log_gamma = nn.Parameter(torch.tensor(0.0))  # gamma = exp(0) = 1
            self._gamma_learn = True
        else:
            self.register_buffer("_log_gamma", torch.tensor(math.log(max(gamma, 1e-6))))
            self._gamma_learn = False
            
        self._stieltjes = StieltjesTransform(q=q, num_iter=num_iter, eps=eps)

    @property
    def gamma(self) -> float:
        return self._log_gamma.exp().item()

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        queries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if queries is None:
            queries = kwargs.get("queries", None)
        if queries is None:
            raise ValueError("AdaptiveScalableStieltjes requires `queries` for beta computation.")

        added_head_dim = False
        if queries.dim() == 3:
            queries = queries.unsqueeze(1)
            added_head_dim = True
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
            added_head_dim = True

        if queries.dim() != 4 or logits.dim() != 4:
            raise ValueError(
                f"AdaptiveScalableStieltjes expects queries/logits 3D or 4D; "
                f"got queries.dim={queries.dim()} logits.dim={logits.dim()}."
            )
        
        K = logits.size(dim if dim >= 0 else (logits.dim() + dim))
        
        # Apply the dot product (starts at 0.0)
        dot_product = torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
        
        # Add the bias (starts at -5.0), broadcasting across batch and sequence length
        beta_logits = dot_product + self.b_beta.view(1, -1, 1)
        
        # softplus(-5.0) safely initializes beta to ~0.0067
        beta = torch.nn.functional.softplus(beta_logits)
        
        gam = self._log_gamma.exp()
        
        # Safely compute log(K) to prevent 0.0 ** gam = NaN gradients during KV cache prefilling
        log_k = math.log(max(float(K), 2.0))
        scale = self.delta + beta * (log_k ** gam)
        scale = scale.unsqueeze(-1)

        # --- FIX: Prevent NaN Gradients from 0 * -inf ---
        is_masked = (logits == float('-inf'))
        safe_logits = torch.where(is_masked, torch.zeros_like(logits), logits)
        
        scaled_logits = scale * safe_logits
        scaled_logits = scaled_logits.masked_fill(is_masked, float('-inf'))
        # ------------------------------------------------
        
        out = self._stieltjes.translate_logits(scaled_logits, dim=dim)

        if added_head_dim and out.size(1) == 1:
            out = out.squeeze(1)
        return out