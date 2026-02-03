from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from asentmax_comp.max_retrieval_architecture.architecture import MaxRetrievalModel
from asentmax_comp.mappings.type_enum import SimplexMappingEnum


@dataclass(frozen=True)
class Experiment:
    name: str
    mapping: SimplexMappingEnum
    mapping_kwargs: Dict


ID_LEN: int = 16
OOD_LENS: List[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096]
ALL_LENS: List[int] = [ID_LEN] + OOD_LENS

# Table 8 rows (names match the prompt).
EXPERIMENTS: List[Experiment] = [
    Experiment("Softmax Veličković et al. (2025)", SimplexMappingEnum.softmax, {}),
    Experiment("Adapt. temp. Veličković et al. (2025)", SimplexMappingEnum.adaptive_temperature, {}),
    # NOTE: for explicit θ experiments, avoid also applying dot-product scaling in the model.
    # Otherwise you effectively scale twice (e.g. /sqrt(d) and then /θ).
    Experiment("Softmax θ = √d", SimplexMappingEnum.softmax, {"temperature": "root_d", "attn_score_scale": "none"}),
    Experiment("Softmax θ = 0.1", SimplexMappingEnum.softmax, {"temperature": 0.1, "attn_score_scale": "none"}),
    Experiment("Softmax θ = 0.0004", SimplexMappingEnum.softmax, {"temperature": 0.0004, "attn_score_scale": "none"}),
    Experiment("SSMax", SimplexMappingEnum.scalable_softmax, {}),
    Experiment("Top-K, K = 2", SimplexMappingEnum.topk_attn, {"k": 2}),
    Experiment("Top-K, K = 4", SimplexMappingEnum.topk_attn, {"k": 4}),
    Experiment("Entmax α = 1.5", SimplexMappingEnum.alpha_entmax, {"alpha": 1.5}),
    Experiment("Entmax α = 2", SimplexMappingEnum.alpha_entmax, {"alpha": 2.0}),
    Experiment("Entmax α = 4", SimplexMappingEnum.alpha_entmax, {"alpha": 4.0}),
    Experiment("Entmax α = 16", SimplexMappingEnum.alpha_entmax, {"alpha": 16.0}),
    Experiment("Entmax α = 32", SimplexMappingEnum.alpha_entmax, {"alpha": 32.0}),
    Experiment("Entmax α = 64", SimplexMappingEnum.alpha_entmax, {"alpha": 64.0}),
    Experiment("ASEntmax, α = 1.5, βlearn, γ = 1", SimplexMappingEnum.as_entmax, {"gamma": 1.0, "delta": 1.0}),
    Experiment("ASEntmax, α = 1.5, βlearn, γ = 2", SimplexMappingEnum.as_entmax, {"gamma": 2.0, "delta": 1.0}),
    Experiment("ASEntmax, α = 1.5, βlearn, γ = 3", SimplexMappingEnum.as_entmax, {"gamma": 3.0, "delta": 1.0}),
    Experiment("ASEntmax, α = 1.5, βlearn, γ = 4", SimplexMappingEnum.as_entmax, {"gamma": 4.0, "delta": 1.0}),
]


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_max_retrieval_batch(
    batch_size: int,
    seq_len: int,
    n_classes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (items, queries, targets):
    - items: (B, T, 1 + n_classes)  [priority + one-hot class]
    - queries: (B, 1)
    - targets: (B,)  (class index of the max-priority item)
    """
    priorities = torch.rand(batch_size, seq_len, device=device)
    classes = torch.randint(0, n_classes, (batch_size, seq_len), device=device)

    argmax_idx = priorities.argmax(dim=1)  # (B,)
    targets = classes.gather(1, argmax_idx.unsqueeze(1)).squeeze(1).long()  # (B,)

    priorities_t = priorities.unsqueeze(-1)  # (B, T, 1)
    classes_t = F.one_hot(classes, n_classes).to(dtype=torch.float32)  # (B, T, C)
    items = torch.cat([priorities_t, classes_t], dim=-1)  # (B, T, 1+C)

    queries = torch.rand(batch_size, 1, device=device)
    return items, queries, targets


def train_max_retrieval(
    model: nn.Module,
    *,
    seq_len: int,
    n_classes: int,
    device: str,
    training_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(training_steps):
        items, queries, targets = sample_max_retrieval_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            n_classes=n_classes,
            device=device,
        )
        opt.zero_grad(set_to_none=True)
        logits = model(items, queries)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()


@torch.no_grad()
def eval_accuracy(
    model: nn.Module,
    *,
    seq_len: int,
    n_classes: int,
    device: str,
    eval_samples: int,
    batch_size: int,
) -> float:
    model.eval()

    correct = 0
    total = 0
    steps = int(np.ceil(eval_samples / batch_size))

    for _ in range(steps):
        items, queries, targets = sample_max_retrieval_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            n_classes=n_classes,
            device=device,
        )
        logits = model(items, queries)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()

    return 100.0 * correct / max(total, 1)


def run_table8() -> None:
    # Defaults (you can tweak to match the paper’s exact config if needed).
    d_emb = 128
    n_classes = 10
    item_input_dim = 1 + n_classes

    training_steps = 20_000
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-3

    eval_samples = 4096

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results: Dict[str, List[float]] = {}
    for exp in tqdm(EXPERIMENTS):
        _set_seeds(0)

        model = MaxRetrievalModel(
            simplex_mapping=exp.mapping,
            d_emb=d_emb,
            n_classes=n_classes,
            item_input_dim=item_input_dim,
            query_input_dim=1,
            attn_score_scale="inv_sqrt_d",
            **exp.mapping_kwargs,
        ).to(device)

        train_max_retrieval(
            model,
            seq_len=ID_LEN,
            n_classes=n_classes,
            device=device,
            training_steps=training_steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
        )

        row: List[float] = []
        for L in ALL_LENS:
            row.append(
                eval_accuracy(
                    model,
                    seq_len=L,
                    n_classes=n_classes,
                    device=device,
                    eval_samples=eval_samples,
                    batch_size=batch_size,
                )
            )
        results[exp.name] = row

    # Table-like TSV output (easy to paste into a spreadsheet).
    header = ["Model"] + [str(L) for L in ALL_LENS]
    print("\t".join(header))
    for name, row in results.items():
        print("\t".join([name] + [f"{x:.1f}" for x in row]))


if __name__ == "__main__":
    run_table8()
