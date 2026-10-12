import torch
import torch.nn.functional as F

from .base_cls import ProbabilitySimplexMapping

class AdaptiveSoftmax(ProbabilitySimplexMapping):
    """Adaptive temperature softmax.
    
    Adapted from: https://arxiv.org/pdf/2410.01104 ; temperature is a function of shannons entropy of logits
    """
    def __init__(self, coeffs=None): #init included for registering
        super().__init__()
        if coeffs is None: coeffs = [-0.037, 0.481, -2.3, 4.917, -1.791]
        self.register_buffer("poly_fit", torch.tensor(coeffs, dtype=torch.float32))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _polyval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x, dtype=torch.float32)
        for c in coeffs: out = out * x + c
        return out

    def translate_logits(self, logits: torch.Tensor, dim: int, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            probs = F.softmax(logits, dim=dim).to(torch.float32)
            log_probs = F.log_softmax(logits, dim=dim).to(torch.float32)
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        poly_fit = self.poly_fit.to(logits.device)
        one = self.one.to(logits.device)
        
        poly_val = self._polyval_horner(poly_fit, entropy)
        greater_mask = entropy > 0.5
        
        poly_val = torch.clamp(poly_val, min=1.0, max=10.0)
        
        beta = torch.where(greater_mask, torch.maximum(poly_val, one), one)
        beta = beta.to(dtype=logits.dtype)
        
        logits_clamped = torch.clamp(logits, min=-50.0, max=50.0)
        logits_scaled = logits_clamped * beta
        
        return F.softmax(logits_scaled, dim=dim)