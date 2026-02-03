import torch
import torch.nn.functional as F
from typing import Optional, Union

from .base_cls import ProbabilitySimplexMapping

class Softmax(ProbabilitySimplexMapping):
    def __init__(self, temperature: Optional[Union[str, float]] = None):
        super().__init__()
        self._temperature = temperature
        self._root_mode = (temperature == 'root_d')

    def translate_logits(
        self,
        logits,
        dim,
        **kwargs,
    ):
        # Implements softmax(logits / theta). If theta="root_d", interpret it as
        # theta = sqrt(d_model) (standard dot-product attention scaling).
        if self._root_mode:
            d_model = kwargs.get("d_model", kwargs.get("d_emb", None))
            if d_model is None:
                raise ValueError(
                    "Softmax(temperature='root_d') requires `d_model`/`d_emb` "
                    "to be passed to translate_logits(...)."
                )
            theta = float(d_model) ** 0.5
            logits = logits / theta
        elif isinstance(self._temperature, (float, int)):
            theta = float(self._temperature)
            logits = logits / theta

        return F.softmax(logits, dim=dim)
