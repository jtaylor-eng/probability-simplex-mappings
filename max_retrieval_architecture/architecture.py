import torch
import torch.nn as nn
import inspect
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
        attn_score_scale: Literal["inv_sqrt_d", "none"] = "inv_sqrt_d",
        **kwargs,  # mapping kwargs (init + forward)
    ):
        super().__init__()
        self._translation_cls = simplex_mapping.value
        self.attn_score_scale = attn_score_scale
        self.d_emb = d_emb

        # Build init kwargs by inspecting the mapping's __init__ signature.
        init_kwargs: dict = {}
        sig = inspect.signature(self._translation_cls.__init__)
        init_param_names = {
            p.name
            for p in sig.parameters.values()
            if p.name not in {"self", "args", "kwargs"}
        }

        # Provide common defaults for methods that require model dims.
        if simplex_mapping == SimplexMappingEnum.as_entmax:
            kwargs.setdefault("d_model", d_emb)
            kwargs.setdefault("n_heads", 1)

        for k, v in kwargs.items():
            if k in init_param_names:
                init_kwargs[k] = v

        self._translate_logits = self._translation_cls(**init_kwargs)
        
        # self.psi_x = MLP(item_input_dim, d_emb, d_emb) #items
        # self.psi_q = MLP(query_input_dim, d_emb, d_emb) #query
        
        self.psi_x = nn.Linear(item_input_dim, d_emb)
        self.psi_q = nn.Linear(query_input_dim, d_emb)
        
        # self.q_proj = nn.Linear(d_emb, d_emb)
        # self.k_proj = nn.Linear(d_emb, d_emb)
        # self.v_proj = nn.Linear(d_emb, d_emb)
        self.q_proj = nn.Linear(query_input_dim, d_emb)
        self.k_proj = nn.Linear(item_input_dim, d_emb)
        self.v_proj = nn.Linear(item_input_dim, d_emb)
        
        # self.phi = MLP(d_emb, d_emb, n_classes)
        self.phi = nn.Linear(d_emb, n_classes)
            
        self.kwargs = kwargs

    def forward(self, x_items, x_query, return_attn=False):
        # x_items (B, T, item_input_dim); x_query (B, 1)
        
        x_query_unsqueezed = x_query.unsqueeze(-1)
        # h_items = self.psi_x(x_items)
        # h_query = self.psi_q(x_query_unsqueezed)
        
        # q = self.q_proj(h_query)
        # k = self.k_proj(h_items)
        # v = self.v_proj(h_items)
        q = self.q_proj(x_query_unsqueezed)
        k = self.k_proj(x_items)
        v = self.v_proj(x_items)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        if self.attn_score_scale == "inv_sqrt_d":
            attn_scores = attn_scores * (self.d_emb ** -0.5)
        elif self.attn_score_scale == "none":
            pass
        else:
            raise ValueError(f"Unknown attn_score_scale={self.attn_score_scale!r}")

        call_kwargs = dict(self.kwargs)
        call_kwargs.pop("d_model", None)
        call_kwargs.pop("d_emb", None)
        call_kwargs.pop("queries", None)

        attn_weights = self._translate_logits.translate_logits(
            attn_scores,
            dim=-1,
            queries=q,
            d_emb=self.d_emb,
            d_model=self.d_emb,
            **call_kwargs,
        )

        z = torch.matmul(attn_weights, v)
        z = z.squeeze(1)
        
        out_logits = self.phi(z)

        return (out_logits, attn_weights) if return_attn else out_logits
    
if __name__ == '__main__':
    print(SimplexMappingEnum)