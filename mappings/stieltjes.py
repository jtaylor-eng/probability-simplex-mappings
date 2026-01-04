import torch

from base_cls import ProbabilitySimplexMapping

class StieltjesTransform(ProbabilitySimplexMapping):
    """Stieltjes transform as introduced, using binary search."""
    def _line_search_bs(self, num_iter, shifted_logits, eps, q, dim, lb, ub):
        for _ in range(num_iter):
            mid = (lb + ub) / 2.0
            
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=eps), -q),
                dim=dim,
                keepdim=True
            ) - 1
            
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)

        return lb, ub

    def translate_logits(
        self,
        logits,
        dim,
        q: float = 1.0,
        num_iter: int = 32,
        eps: float = 1e-9,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates 1 / (lambda_q - x_i)^q"""
        
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

        lb = torch.full_like(x_max, eps)
        ub = torch.full_like(x_max, logits.shape[dim] ** (1.0/q))

        lb, ub = self._line_search_bs(
            num_iter=num_iter,
            shifted_logits=x_i,
            eps=eps,
            q=q,
            dim=dim,
            lb=lb,
            ub=ub
        )
        lambda_1 = (lb + ub) / 2.0
        
        # 1 / (lambda_q - x_i)^q
        return torch.pow((lambda_1 - x_i).clamp(min=eps), -q)