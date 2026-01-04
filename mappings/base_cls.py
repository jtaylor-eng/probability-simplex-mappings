#Extend this class to create new simplex mappings

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ProbabilitySimplexMapping(nn.Module, ABC):
    """Provide uniform interface to use for probabilistic simplex mappings"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def translate_logits(
        self,
        logits: torch.Tensor,
        dim: int,
        **kwargs,
    ) -> torch.Tensor:
        pass