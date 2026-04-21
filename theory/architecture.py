from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from asentmax_comp.mappings.type_enum import SimplexMappingEnum

DEVICE = 'cuda'

@dataclass
class TransformerParams:
    simplex_mapping: SimplexMappingEnum
    n_heads: int
    n_layers: int
    hidden_dim: int
    int_dim: int
    vocab_size: int
    seq_len: int
    use_nape: bool = True

class MHA(nn.Module):
    def __init__(self, config: TransformerParams):
        super().__init__()
        mapping_class = config.simplex_mapping.value
        
        # We must explicitly pass head_dim and n_heads to Adaptive Scalable Stieltjes
        if mapping_class.__name__ == "AdaptiveScalableStieltjes":
            head_dim = config.hidden_dim // config.n_heads
            self._simplex_mapping = mapping_class(d_model=head_dim, n_heads=config.n_heads)
        else:
            self._simplex_mapping = mapping_class()
            
        self._kvq_projections = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=True)
        self._out_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self._sqrt_dk = math.sqrt(config.hidden_dim // config.n_heads)
        self._config = config

        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                        .view(1, 1, config.seq_len, config.seq_len))
        if config.use_nape:
            self.register_buffer("nape_bias", self._generate_nape_mask(config.seq_len))

    def forward(self, X, use_cache=False, past_kv=None):
        B, T, C = X.size()

        Q, K, V = self._kvq_projections(X).split(self._config.hidden_dim, dim=2)
        K = K.view(B, T, self._config.n_heads, C // self._config.n_heads).transpose(1, 2)
        Q = Q.view(B, T, self._config.n_heads, C // self._config.n_heads).transpose(1, 2)
        V = V.view(B, T, self._config.n_heads, C // self._config.n_heads).transpose(1, 2)

        # --- KV CACHE INJECTION ---
        if use_cache and past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)

        seq_len = K.size(2)

        if seq_len > self.bias.size(-1):
            self.bias = torch.tril(torch.ones(seq_len, seq_len, device=X.device)).view(1, 1, seq_len, seq_len)
            if self._config.use_nape:
                self.nape_bias = self._generate_nape_mask(seq_len).to(X.device)

        att = (Q @ K.transpose(-2, -1)) / self._sqrt_dk
        
        # Slice masks to align with current query (T) and total keys (seq_len)
        if self._config.use_nape:
            att = att + self.nape_bias[:, :, seq_len-T:seq_len, :seq_len]

        att = att.masked_fill(self.bias[:,:,seq_len-T:seq_len,:seq_len] == 0, float('-inf'))
        
        # --- FIXED KWARG INJECTION FOR ADAPTIVE STIELTJES ---
        att = self._simplex_mapping.translate_logits(
            att, dim=-1, queries=Q, d_model=self._config.hidden_dim
        )
        
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self._out_projection(y)

        return (out, (K, V)) if use_cache else out

    def _generate_nape_mask(self, T):
        n_alibi_heads = self._config.n_heads // 2
        slopes = torch.tensor([2 ** (-8.0 / n_alibi_heads * i) for i in range(1, n_alibi_heads + 1)], device=DEVICE)
        
        positions = torch.arange(T, device=DEVICE)
        rel_pos = positions.view(1, T) - positions.view(T, 1)
        rel_pos = torch.clamp(rel_pos, max=0)

        alibi_bias = slopes.view(-1, 1, 1) * rel_pos
        nope_bias = torch.zeros(self._config.n_heads - n_alibi_heads, T, T, device=DEVICE)
        
        return torch.cat([nope_bias, alibi_bias], dim=0).unsqueeze(0)

class MLP(nn.Module):
    def __init__(self, config: TransformerParams):
        super().__init__()
        self.c_fc    = nn.Linear(config.hidden_dim, config.int_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.int_dim, config.hidden_dim)

    def forward(self, X):
        X = self.c_fc(X)
        X = self.gelu(X)
        return self.c_proj(X)

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerParams):
        super().__init__()
        self._ln_1  = nn.LayerNorm(config.hidden_dim)
        self._ln_2  = nn.LayerNorm(config.hidden_dim)
        self._mha = MHA(config)
        self._mlp = MLP(config)
    
    def forward(self, X, use_cache=False, past_kv=None):
        if use_cache:
            mha_out, new_kv = self._mha(self._ln_1(X), use_cache=True, past_kv=past_kv)
            X = X + mha_out
            X = X + self._mlp(self._ln_2(X))
            return X, new_kv
        else:
            X = X + self._mha(self._ln_1(X))
            X = X + self._mlp(self._ln_2(X))
            return X

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerParams):
        super().__init__()
        if not config.use_nape:
            self._pos_embeddings = nn.Embedding(config.seq_len, config.hidden_dim)

        self._tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self._transformer    = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self._ln             = nn.LayerNorm(config.hidden_dim)
        self._lm_head        = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        self._tok_embeddings.weight = self._lm_head.weight
        self._config = config

    def forward(self, X, use_cache=False, past_kvs=None):
        _, T = X.size()
        offset = past_kvs[0][0].size(2) if (use_cache and past_kvs is not None) else 0

        X = self._tok_embeddings(X)

        if not self._config.use_nape:
            pos = torch.arange(offset, offset + T, dtype=torch.long, device=X.device)
            pos_emb = self._pos_embeddings(pos)
            X = X + pos_emb

        new_kvs = []
        for i, layer in enumerate(self._transformer):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if use_cache:
                X, new_kv = layer(X, use_cache=True, past_kv=past_kv)
                new_kvs.append(new_kv)
            else:
                X = layer(X)
        
        X = self._ln(X)
        logits = self._lm_head(X)
        
        return (logits, new_kvs) if use_cache else logits