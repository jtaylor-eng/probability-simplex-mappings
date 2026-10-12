import torch
import torch.nn as nn
from .base_cls import ProbabilitySimplexMapping


class LearnableStieltjes(ProbabilitySimplexMapping):
    def __init__(self, initial_q=2.0):
        super().__init__()
        self.raw_q = nn.Parameter(torch.tensor(float(initial_q)))

    def translate_logits(
        self,
        logits,
        dim,
        num_iter: int = 16,
        eps: float = 1e-7,
        newton_steps: int = 1,
        **kwargs
    ) -> torch.Tensor:
        seq_len = logits.shape[dim]

        len_factor = torch.log2(torch.tensor(seq_len, device=logits.device)) / torch.log2(torch.tensor(10.0))
        q = torch.clamp(self.raw_q * len_factor, min=2, max=14.0)
        
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max


        lb = torch.zeros_like(x_max) + eps
        ub_val = (logits.shape[dim] ** (1.0 / q)) + eps
        ub = torch.ones_like(x_max) * ub_val

        for _ in range(num_iter):
            mid = (lb + ub) / 2.0
            prob_sum = torch.sum(torch.pow((mid - x_i).clamp(min=eps), -q), dim=dim, keepdim=True)
            lb = torch.where(prob_sum > 1.0, mid, lb)
            ub = torch.where(prob_sum <= 1.0, mid, ub)
        
        lambda_val = (lb + ub) / 2.0

        for _ in range(newton_steps):
            diff = lambda_val - x_i
            f = torch.sum(torch.pow(diff.clamp(min=eps), -q), dim=dim, keepdim=True) - 1.0
            f_prime = -q * torch.sum(torch.pow(diff.clamp(min=eps), -(q + 1)), dim=dim, keepdim=True)
            lambda_val = lambda_val - f / f_prime

        return torch.pow((lambda_val - x_i).clamp(min=eps), -q)
