import torch
import torch.nn as nn
from typing import Literal, Optional, Callable, List

from asentmax_comp.mappings.type_enum import SimplexMappingEnum

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class MaxRetrievalModel(nn.Module):
    def __init__(
        self,
        simplex_mapping: SimplexMappingEnum,
        d_emb: int,
        n_classes: int,
        item_input_dim: int,
        query_input_dim: int = 1,
        **kwargs #additional args to be used in construction simplex mapping obj
    ):
        super().__init__()
        self._translation_cls = simplex_mapping.value
        # Only pass kwargs that the mapping's __init__ accepts
        # Most mappings (Softmax, StieltjesTransform) don't accept kwargs in __init__
        # AdaptiveSoftmax accepts 'coeffs' in __init__, so filter kwargs appropriately
        init_kwargs = {}
        if 'coeffs' in kwargs:
            init_kwargs['coeffs'] = kwargs['coeffs']
        self._translate_logits = self._translation_cls(**init_kwargs)
        self.d_emb = d_emb
        
        self.psi_x = MLP(item_input_dim, d_emb, d_emb) #items
        self.psi_q = MLP(query_input_dim, d_emb, d_emb) #query
        
        self.q_proj = nn.Linear(d_emb, d_emb)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        
        self.phi = MLP(d_emb, d_emb, n_classes)
            
        self.kwargs = kwargs

    def forward(self, x_items, x_query, return_attn=False):
        # x_items (B, T, item_input_dim); x_query (B, 1)
        
        x_query_unsqueezed = x_query.unsqueeze(-1)
        h_items = self.psi_x(x_items)
        h_query = self.psi_q(x_query_unsqueezed)
        
        q = self.q_proj(h_query)
        k = self.k_proj(h_items)
        v = self.v_proj(h_items)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        attn_scores *= k.size(-1) ** -0.5 

        attn_weights = self._translate_logits.translate_logits(attn_scores, dim=-1, **self.kwargs)

        z = torch.matmul(attn_weights, v)
        z = z.squeeze(1)
        
        out_logits = self.phi(z)

        return (out_logits, attn_weights) if return_attn else out_logits
    
if __name__ == '__main__':
    print(SimplexMappingEnum)