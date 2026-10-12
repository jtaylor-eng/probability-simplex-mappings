import torch
import torch.nn as nn
import math
from entmax import entmax15
from .base_cls import ProbabilitySimplexMapping

class AdaptiveScalableEntmax(ProbabilitySimplexMapping):
    def __init__(self, d_model, n_heads, gamma=1.0, delta=1.0):
        super().__init__()
        self.delta = delta

        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
        self.w_gamma = gamma

    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        queries: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        ASEntmax scaling (paper Eq. 8-style):
          alpha-entmax((delta + beta * (log K)^gamma) * logits)

        Notes:
        - `queries` can be (B, Q, D) (single head) or (B, H, Q, D).
        - `logits` can be (B, Q, K) or (B, H, Q, K).
        """
        if queries is None:
            queries = kwargs.get("queries", None)
        if queries is None:
            raise ValueError("ASEntmax requires `queries` for beta computation.")

        # Normalize shapes to include head dimension if absent.
        added_head_dim = False
        if queries.dim() == 3:  # (B, Q, D)
            queries = queries.unsqueeze(1)  # (B, 1, Q, D)
            added_head_dim = True
        if logits.dim() == 3:  # (B, Q, K)
            logits = logits.unsqueeze(1)  # (B, 1, Q, K)
            added_head_dim = True

        if queries.dim() != 4 or logits.dim() != 4:
            raise ValueError(
                f"ASEntmax expects queries/logits to be 3D or 4D; "
                f"got queries.dim={queries.dim()} logits.dim={logits.dim()}."
            )

        K = logits.size(dim if dim >= 0 else (logits.dim() + dim))
        beta = torch.nn.functional.softplus(
            torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
        )  # (B, H, Q)

        scale = self.delta + beta * (math.log(K) ** float(self.w_gamma))
        scale = scale.unsqueeze(-1)  # (B, H, Q, 1)

        scaled_logits = scale * logits
        out = entmax15(scaled_logits, dim=dim)

        # If we inserted a head dimension, remove it for compatibility.
        if added_head_dim and out.size(1) == 1:
            out = out.squeeze(1)
        return out
