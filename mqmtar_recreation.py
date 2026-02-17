from typing import Dict, List
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn

from zoology.logger import WandbLogger
from zoology.model import LanguageModel
from zoology.utils import set_determinism
from zoology.train import Trainer
from zoology.data.utils import prepare_data

from asentmax_comp.mappings.type_enum import SimplexMappingEnum
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig, ModuleConfig
from zoology.data.multiquery_ar import MQARConfig

@dataclass
class Experiment:
    name: str
    mapping: SimplexMappingEnum
    mapping_kwargs: Dict

ID_LEN: int = 64
OOD_LEN_MULTIPLES: List[int] = [2]
STIELTJES_QS: List[int] = [2, 4, 8, 16, 32, 64]

EXPERIMENTS: List[Experiment] = [
    *[Experiment(f"Traditional Stieltjes q={q}", SimplexMappingEnum.stieltjes, {'q': float(q)}) for q in STIELTJES_QS],
    Experiment('Softmax', SimplexMappingEnum.softmax, {}),
    Experiment('Sparsemax', SimplexMappingEnum.sparsemax, {}),
    Experiment('Top-K (K=32)', SimplexMappingEnum.topk_attn, {'k': 32}),
    Experiment('Alpha Entmax (alpha=1.5)', SimplexMappingEnum.alpha_entmax, {}),
    # Experiment('Adaptive Scalable Entmax (alpha=1.5)', SimplexMappingEnum.as_entmax, {}),
]

class CustomSimplexMappingAttention(nn.Module):
    def __init__(self, simplex_mapping_name: str, d_model: int, n_heads: int = 1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        kwargs.pop('layer_idx', None)
        mapping_class = SimplexMappingEnum[simplex_mapping_name].value

        sig = inspect.signature(mapping_class.__init__)
        init_params = sig.parameters

        mapping_kwargs = {}
        if 'd_model' in init_params:
            mapping_kwargs['d_model'] = d_model

        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in init_params.values())

        for k, v in kwargs.items():
            if k in init_params or accepts_kwargs:
                mapping_kwargs[k] = v

        self.mapping_instance = mapping_class(**mapping_kwargs)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, l, d = x.shape

        q = self.W_q(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.W_k(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, -float('inf'))

        attn_weights = self.mapping_instance.translate_logits(
            logits=scores,
            dim=-1,
            d_model=self.head_dim,
            queries=q
        )

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, l, d)

        return self.W_o(out)

def create_experiment_configs(experiments: List[Experiment], id_len: int, ood_len_multiples: List[int]):
    max_seq_len = id_len * max(ood_len_multiples)
    safe_vocab_size = max_seq_len + 256

    configs = []

    test_configs = [
        MQARConfig(
            name='mqar_id',
            vocab_size=safe_vocab_size,
            input_seq_len=id_len,
            num_examples=1000,
            include_slices=False
        )
    ]
    for m in ood_len_multiples:
        test_configs.append(
            MQARConfig(
                name=f"mqar_{m}x",
                vocab_size=safe_vocab_size,
                input_seq_len=id_len * m,
                num_examples=1000,
                include_slices=False
            )
        )

    data_config = DataConfig(
        train_configs=[
            MQARConfig(
                vocab_size=safe_vocab_size,
                input_seq_len=id_len,
                num_examples=100_000,
                include_slices=False
            )
        ],
        test_configs=test_configs,
        batch_size=256,
        cache_dir='./data_cache_mqar'
    )

    for exp in experiments:
        kwargs = exp.mapping_kwargs.copy()
        kwargs['simplex_mapping_name'] = exp.mapping.name
        kwargs['n_heads'] = 2

        model_config = ModelConfig(
            d_model=64,
            n_layers=4,
            vocab_size=safe_vocab_size + 8,
            max_position_embeddings=max_seq_len + 256,
            sequence_mixer=ModuleConfig(
                name='asentmax_comp.mqmtar_recreation.CustomSimplexMappingAttention',
                kwargs=kwargs
            )
        )

        train_config = TrainConfig(
            data=data_config,
            model=model_config,
            max_epochs=10,
            learning_rate=3e-4,
            logger=LoggerConfig(project_name='asentmax_test', log_to_wandb=False)
        )

        configs.append((exp, train_config))

    return configs

def run_experiment(config: TrainConfig):
    set_determinism(config.seed)
    logger = WandbLogger(config)

    train_data_config = config.data.model_copy(deep=True)
    train_data_config.batch_size = 256
    train_loader, test_loader = prepare_data(train_data_config)

    model = LanguageModel(config.model)
    logger.log_model(model, config=config)
    print(model)

    task = Trainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    task.fit()

    final_metrics = {}

    for test_cfg in config.data.test_configs:
        temp_data_config = config.data.model_copy(deep=True)
        temp_data_config.test_configs = [test_cfg]
        temp_data_config.batch_size = 16

        _, specialized_test_loader = prepare_data(temp_data_config)
        task.test_dataloader = specialized_test_loader

        metrics = task.test(epoch_idx=config.max_epochs)
        acc = metrics.get('valid/accuracy', 0.0)
        final_metrics[f"valid/{test_cfg.name}/accuracy"] = acc

        print(f"  -> {test_cfg.name}: {acc*100:.2f}%")

    logger.finish()
    return final_metrics

def print_results(results_list, ood_len_multiples):
    print('\n' + '='*120)
    header = f"{'Method':<35} | {'ID':<8} | " + " | ".join([f"{m}x".ljust(8) for m in ood_len_multiples])
    print(header)
    print('-' * 120)

    for exp_name, metrics in results_list:
        if metrics is None:
            print(f"{exp_name:<35} | ERROR: No metrics returned")
            continue

        id_acc = metrics.get('valid/mqar_id/accuracy', 0.0) * 100
        ood_accs = []

        for m in ood_len_multiples:
            acc = metrics.get(f"valid/mqar_{m}x/accuracy", 0.0) * 100
            ood_accs.append(f"{acc:.1f}")

        row_str = f"{exp_name:<35} | {id_acc:.1f}     | " + " | ".join([str(a).ljust(8) for a in ood_accs])
        print(row_str)

    print('='*120 + '\n')

def main():
    torch.cuda.empty_cache()

    experiment_configs = create_experiment_configs(
        EXPERIMENTS,
        ID_LEN,
        OOD_LEN_MULTIPLES
    )
    results = []

    print(f"Running {len(experiment_configs)} Experiments...")

    for i, (exp, config) in enumerate(experiment_configs):
        print(f"\n[{i+1}/{len(experiment_configs)}] Running Experiment: {exp.name}")
        metrics = run_experiment(config)
        results.append((exp.name, metrics))

    print_results(results, OOD_LEN_MULTIPLES)

    return 0

if __name__ == '__main__':
    exit(main())
