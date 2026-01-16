# Experiment running goes here
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

from simplex_mappings.max_retrieval_architecture.architecture import MaxRetrievalModel
from simplex_mappings.mappings.type_enum import SimplexMappingEnum
from simplex_mappings.dataset_gen.gen import make_dataset


def plot_max_retrieval_attention(
    model, 
    device, 
    save_path, 
    n_classes, 
    item_input_dim, 
    start_len=16, 
    num_doubles=8, 
    batch_size=32
):
    """
    Plots attention maps like Figure 2 from the paper [cite: 108-110].
    """
    model.eval()
    fig, axes = plt.subplots(1, num_doubles, figsize=(15, 5), sharey=True)
    fig.subplots_adjust(wspace=0.1)

    current_len = start_len
    for i in range(num_doubles):
        ax = axes[i]
        
        priorities = torch.rand(batch_size, current_len).to(device)
        classes = torch.randint(0, n_classes, (batch_size, current_len)).to(device)
        
        priorities_t = priorities.unsqueeze(-1)
        classes_t = F.one_hot(classes, n_classes).float()
        items = torch.cat([priorities_t, classes_t], dim=-1)  # (B, T, 1+C)
        
        query_vec = torch.rand(batch_size, 1).to(device)  # (B, 1)

        with torch.no_grad():
            _, attn_weights = model(items, query_vec, return_attn=True)
            attn_weights = attn_weights.squeeze(1)  # (B, 1, T) -> (B, T)

        # by priority
        sorted_value_indices = torch.argsort(priorities, dim=1)
        top_16_value_indices = sorted_value_indices[:, -16:]
        top_k_weights = torch.gather(attn_weights, 1, top_16_value_indices)
        
        ax.imshow(top_k_weights.cpu().numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"{current_len}", fontsize=8)
        ax.set_xticks([])
        
        current_len *= 2

    plt.savefig(f'{save_path}.png')
    plt.close()


def plot_learning_curve(
    train_losses: List[float],
    val_ID_losses: List[float],
    val_OOD_losses: List[float],
    train_time: float,
    name: str,
    save_path: str,
    log_every_steps: int
):
    steps_range = list(range(1, len(train_losses) + 1))
    steps_ticks = [s * log_every_steps for s in steps_range]

    plt.figure()
    plt.plot(steps_ticks, train_losses, 'b-o', label='Training Loss')
    plt.plot(steps_ticks, val_ID_losses, 'g-o', label='In-Distribution Val Loss')
    plt.plot(steps_ticks, val_OOD_losses, 'r-o', label='OOD Val Loss')
    plt.title(f'Learning curves for {name} | trained for {train_time:.2f} seconds.')
    plt.legend()
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (normalized)')
    plt.grid(True)
    plt.savefig(f'{save_path}.png')
    plt.close()


def train_or_val(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    train_mode: bool = True,
    optim: Optional[Optimizer] = None,
) -> float:
    """Run training or validation on model given other args."""
    total_loss = 0.0
    
    if train_mode:
        assert optim, 'need optimizer for train mode'
        model.train()
        
        for items, queries, targets in dataloader:
            items = items.to(device)
            queries = queries.to(device)
            targets = targets.to(device)
            
            optim.zero_grad()
            
            out_logits = model(items, queries)
            loss = loss_fn(out_logits, targets)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
    
    else:
        model.eval()
        with torch.no_grad(): 
            for items, queries, targets in dataloader:
                items = items.to(device)
                queries = queries.to(device)
                targets = targets.to(device)
                
                out_logits = model(items, queries)
                loss = loss_fn(out_logits, targets)
                total_loss += loss.item()
                
    return total_loss / len(dataloader)


if __name__ == '__main__':
    results_folder = './results/'
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # hyperparams
    d_emb = 128
    n_classes = 10
    training_steps = 20000
    log_every_steps = 500
    batch_size = 128
    max_len_seq = 16
    min_len_seq = 5
    ood_len_seq = 128  # OOD validation
    lr = 0.001
    weight_decay = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataloaders
    item_input_dim = 1 + n_classes
    len_train_dataset = training_steps * batch_size 
    len_val_dataset = 1024
    
    dataloader = make_dataset(
        len_train_dataset, 
        max_len_seq, 
        n_classes, 
        min_len_seq, 
        batch_size=batch_size
    )
    dataloader_val_ID = make_dataset(
        len_val_dataset, 
        max_len_seq, 
        n_classes, 
        min_len_seq, 
        batch_size=batch_size
    )
    dataloader_val_OOD = make_dataset(
        len_val_dataset, 
        ood_len_seq, 
        n_classes, 
        min_len_seq, 
        batch_size=batch_size
    )
    
    train_iter = iter(dataloader)  # step training

    # Test different simplex mappings
    for logits_translation in [SimplexMappingEnum.softmax, SimplexMappingEnum.stieltjes, SimplexMappingEnum.adaptive_temperature]:
        start = time.time()

        # setup
        model = MaxRetrievalModel(
            simplex_mapping=logits_translation,
            d_emb=d_emb,
            n_classes=n_classes,
            item_input_dim=item_input_dim,
            query_input_dim=1,
            q=8  # for stieltjes
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        # step training
        train_losses, val_ID_losses, val_OOD_losses = [], [], []
        
        for step in tqdm(range(training_steps), desc=f"Training {logits_translation.name}"):
            try:
                items, queries, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                items, queries, targets = next(train_iter)
            
            items, queries, targets = items.to(device), queries.to(device), targets.to(device)
            
            model.train()
            optimizer.zero_grad()
            out_logits = model(items, queries)
            loss = loss_fn(out_logits, targets)
            loss.backward()
            optimizer.step()
            
            # log
            if step > 0 and (step % log_every_steps == 0 or step == training_steps - 1):
                val_ID_loss = train_or_val(
                    model, 
                    dataloader_val_ID, 
                    loss_fn, 
                    device, 
                    train_mode=False
                )
                val_OOD_loss = train_or_val(
                    model, 
                    dataloader_val_OOD, 
                    loss_fn, 
                    device, 
                    train_mode=False
                )
                
                train_losses.append(loss.item())
                val_ID_losses.append(val_ID_loss)
                val_OOD_losses.append(val_OOD_loss)
                
                print(f'\nStep {step} | Train Loss: {train_losses[-1]:.4f} | Val Loss (ID): {val_ID_losses[-1]:.4f} | Val Loss (OOD): {val_OOD_losses[-1]:.4f}')
        
        # plotting
        time_train = time.time() - start
        print(f'Time to train using {logits_translation.name}: {time_train}')
        
        plot_max_retrieval_attention(
            model=model,
            device=device,
            save_path=results_folder + logits_translation.name,
            n_classes=n_classes,
            item_input_dim=item_input_dim
        )

        plot_learning_curve(
            train_losses=train_losses,
            val_ID_losses=val_ID_losses,
            val_OOD_losses=val_OOD_losses,
            train_time=time_train,
            name=logits_translation.name,
            save_path=results_folder + f'{logits_translation.name}_learning_curves',
            log_every_steps=log_every_steps
        )
