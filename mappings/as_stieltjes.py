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
        d_model: int,
        n_heads: int = 1,
        gamma: float | None = None,
        delta: float = 1.0,
        q: float = 2.0,
        num_iter: int = 16,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.delta = delta
        self.w_beta = nn.Parameter(torch.zeros(n_heads, d_model))
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
        d_emb: int | None = None,
        d_model: int | None = None,
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
        beta = torch.nn.functional.softplus(
            torch.einsum("bhqd,hd->bhq", queries, self.w_beta)
        )
        gam = self._log_gamma.exp()
        scale = self.delta + beta * (math.log(float(K)) ** gam)
        scale = scale.unsqueeze(-1)

        scaled_logits = scale * logits
        out = self._stieltjes.translate_logits(scaled_logits, dim=dim)

        if added_head_dim and out.size(1) == 1:
            out = out.squeeze(1)
        return out
