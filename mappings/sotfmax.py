import torch.nn.functional as F

from base_cls import ProbabilitySimplexMapping

class Softmax(ProbabilitySimplexMapping):
    """Just torch.nn.functional.softmax"""
    def translate_logits(self, logits, dim, **kwargs): 
        return F.softmax(logits, dim=dim)
