import torch

from .base_cls import ProbabilitySimplexMapping

class StieltjesTransform(ProbabilitySimplexMapping):
    """Stieltjes / Tsallis-q simplex mapping via bisection on the dual variable.

    yᵢ = (λ_q - xᵢ)^(-q),  Σ yᵢ = 1
    Upper bound n^(1/q) is dynamic so OOD lengths beyond training length work.
    Output is explicitly normalised to guard against bisection float drift.
    """
    def __init__(
        self,
        q: float = 1.0,
        num_iter: int = 16,
        eps: float = 1e-9
    ):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _bisect_lambda(self, shifted_logits, dim, lb, ub):
        for _ in range(self._num_iter):
            mid = (lb + ub) * 0.5
            f_mid = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
                dim=dim,
                keepdim=True,
            ) - 1.0
            lb = torch.where(f_mid > 0.0, mid, lb)
            ub = torch.where(f_mid <= 0.0, mid, ub)
        return (lb + ub) * 0.5

    def translate_logits(
        self,
        logits,
        dim,
        **kwargs,
    ) -> torch.Tensor:

        n = logits.shape[dim]
        x_max = logits.max(dim=dim, keepdim=True).values
        shifted = logits - x_max

        lb = torch.full_like(x_max, self._eps)
        ub = torch.full_like(x_max, n ** (1.0 / self._q))

        lam = self._bisect_lambda(shifted, dim, lb, ub)

        probs = torch.pow((lam - shifted).clamp(min=self._eps), -self._q)
        return probs / probs.sum(dim=dim, keepdim=True).clamp(min=self._eps)