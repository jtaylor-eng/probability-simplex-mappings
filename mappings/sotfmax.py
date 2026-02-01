import torch
import torch.nn.functional as F
from typing import Optional, Union

from .base_cls import ProbabilitySimplexMapping

class Softmax(ProbabilitySimplexMapping):
    def translate_logits(
        self,
        logits,
        dim,
        temperature: Optional[Union[str, float]] = None,
        **kwargs,
    ):
        if temperature == 'root_d':
            logits /= torch.sqrt(torch.tensor(dim))
        if type(temperature) == float:
            logits /= temperature
        
        return F.softmax(logits, dim=dim)
