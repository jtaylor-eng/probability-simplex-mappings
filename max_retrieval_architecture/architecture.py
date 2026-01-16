import torch
import torch.nn as nn
from typing import Literal, Optional, Callable, List

from simplex_mappings.mappings.type_enum import SimplexMappingEnum

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
        d_emb: int,
        n_classes: int,
        item_input_dim: int,
        query_input_dim: int = 1,
        softmax: Literal['traditional', 'stieltjes', 'adaptive'] = 'traditional',
        **kwargs # To pass q to stieltjes
    ):
        super().__init__()
        self._translation_name = softmax
        self.d_emb = d_emb
        
        self.psi_x = MLP(item_input_dim, d_emb, d_emb) #items
        self.psi_q = MLP(query_input_dim, d_emb, d_emb) #query
        
        self.q_proj = nn.Linear(d_emb, d_emb)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        
        self.phi = MLP(d_emb, d_emb, n_classes)
        
        if softmax == 'traditional': self._translate_logits = TraditionalSoftmax()
        elif softmax == 'stieltjes': self._translate_logits = StieltjesTransform()
        elif softmax == 'adaptive': self._translate_logits = AdaptiveSoftmax()
        else: raise ValueError('Error: Invalid softmax option.')
            
        self.kwargs = kwargs

    def forward(self, x_items, x_query, return_attn=False):
        # x_items (B, T, item_input_dim); x_query (B, 1)
        
        x_query_unsqueezed = x_query.unsqueeze(-1) # (B, 1) -> (B, 1, 1)
        h_items = self.psi_x(x_items)  # (B, T, d_emb)
        h_query = self.psi_q(x_query_unsqueezed)  # (B, 1, d_emb)
        
        q = self.q_proj(h_query) # (B, 1, d_emb)
        k = self.k_proj(h_items) # (B, T, d_emb)
        v = self.v_proj(h_items) # (B, T, d_emb)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self._translation_name == 'traditional': #needed for trad
            attn_scores *= k.size(-1) ** -0.5 

        attn_weights = self._translate_logits.translate_logits(
            attn_scores, 
            dim=-1,
            **self.kwargs
        ) # (B, 1, T)

        z = torch.matmul(attn_weights, v)
        z = z.squeeze(1) # (B, d_emb)
        
        out_logits = self.phi(z) # (B, n_classes)

        return (out_logits, attn_weights) if return_attn else out_logits
    
if __name__ == '__main__':
    print(SimplexMappingEnum) ; exit()
