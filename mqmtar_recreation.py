from typing import Dict, List
from dataclasses import dataclass
import torch
import torch.nn as nn

from asentmax_comp.mappings.type_enum import SimplexMappingEnum

from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, FunctionConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig
from zoology.train import train


@dataclass
class Experiment:
    name: str
    mapping: SimplexMappingEnum
    mapping_kwargs: Dict


ID_LEN: int= 64
OOD_LEN_MULTIPLES: List[int] = [2, 4, 16, 64, 256, 1024]

STIELTJES_Q_LENS: List[int] = [2,4,8,16,32,64]

EXPERIMENTS: List[Experiment] = [
    *[Experiment(f"Adaptive Temperature Stieltjes q={q}",SimplexMappingEnum.adaptive_temperature_stieltjes,{"q": float(q)},) for q in STIELTJES_Q_LENS],
    *[Experiment(f"Traditional Stieltjes q={q}",SimplexMappingEnum.stieltjes,{"q": float(q)},) for q in STIELTJES_Q_LENS],
    Experiment("Softmax", SimplexMappingEnum.softmax, {}),
    Experiment("Sparsemax", SimplexMappingEnum.sparsemax, {}),
    Experiment("Top-K (K=32)", SimplexMappingEnum.topk_attn, {"k", 32}),
    Experiment("Alpha Entmax (alpha=1.5)", SimplexMappingEnum.alpha_entmax, {}),
    Experiment("Adaptive Scalable Entmax (alpha=1.5)", SimplexMappingEnum.alpha_entmax, {}), #TODO: fix d_emb arg passing
]

class CustomSimplexMappingAttention(nn.Module):
    def __init__(
        self,
        simplex_mapping_name: str, 
        d_model: int,
        n_heads: int = 1,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        kwargs.pop('layer_idx', None)
        mapping_enum = SimplexMappingEnum[simplex_mapping_name]
        self.mapping_instance = mapping_enum.value(**kwargs)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        
        q = self.W_q(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, -float('inf'))
        
        attn_weights = self.mapping_instance.translate_logits(
            logits=scores, 
            dim=-1, 
            d_model=self.d_model
        )
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        
        return self.W_o(out)


def create_experiments(
    experiments: List[Experiment],
    id_len: int,
    oud_len_multiples: List[int],
):
    ood_lens = [id_len * ood_len for ood_len in oud_len_multiples]
    max_len = max(ood_lens)
    #TODO use same data for each experiment (slow to gen 100k datapoints)
    data = DataConfig(
        train_configs=[MQARConfig(vocab_size=max_len, input_seq_len=id_len, num_examples=1000000, include_slices=False)],
        test_configs=[*[MQARConfig(vocab_size=max_len, input_seq_len=curr_size, num_examples=1000, include_slices=False) for curr_size in ood_lens]],
        batch_size=256
    )

    #TODO support the list of experiments rather than just 1
    model = ModelConfig(
        d_model=64,
        n_layers=4,
        vocab_size=max_len + 8, #+8 to be save for reserved tokens?
        max_position_embeddings=512,
        sequence_mixer=ModuleConfig(
            name="asentmax_comp.mqmtar_recreation.CustomSimplexMappingAttention",
            kwargs= {
                "simplex_mapping_name": "as_entmax",
                "n_heads": 2,
                # "d_model": 64,
                # "q": 16
            } 
        )
    )

    config = TrainConfig(
        data=data,
        model=model,
        max_epochs=1,
        learning_rate=1e-3,
        logger=LoggerConfig(project_name="asentmax_test", log_to_wandb=False)
    )
    
    return config


def print_results(results_list, ood_len_multiples):
    print("\n" + "="*100)
    print(f"{'Method':<35} | {'ID':<8} | " + " | ".join([f"{m}x".ljust(8) for m in ood_len_multiples]))
    print("-" * 100)

    for exp_name, metrics in results_list:
        id_acc = metrics.get("valid/mqar_id/accuracy", 0.0) * 100
        ood_accs = []

        for m in ood_len_multiples:
            acc = metrics.get(f"valid/mqar_{m}x/accuracy", 0.0) * 100
            ood_accs.append(f"{acc:.1f}")
            
        row_str = f"{exp_name:<35} | {id_acc:.1f}     | " + " | ".join([str(a).ljust(8) for a in ood_accs])
        print(row_str)

    print("="*100 + "\n")


def main():
    experiments = create_experiments(
        EXPERIMENTS,
        ID_LEN,
        OOD_LEN_MULTIPLES
    )

    res = train(experiments)
    print_results(res)

    return 0

if __name__ == '__main__':
    exit(main())