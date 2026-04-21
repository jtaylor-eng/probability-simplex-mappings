import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from asentmax_comp.mappings.type_enum import SimplexMappingEnum
from asentmax_comp.theory.architecture import DecoderOnlyTransformer, TransformerParams

class Seq2SeqTaskDataset(Dataset):
    def __init__(self, src_file, trg_file, sep_token=1):
        self.sep_token = sep_token

        print(f"Scanning dataset to allocate memory: {src_file}")
        
        # --- PASS 1: Fast scan for dimensions (uses near zero RAM) ---
        num_lines = 0
        max_s_len = 0
        max_t_len = 0
        
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for s_line, t_line in zip(f_src, f_trg):
                num_lines += 1
                # Fast way to count tokens without full string splitting
                max_s_len = max(max_s_len, s_line.count(' ') + 1)
                max_t_len = max(max_t_len, t_line.count(' ') + 1)

        print(f"Found {num_lines} samples. Pre-allocating contiguous tensors...")

        # --- PRE-ALLOCATE CONTIGUOUS MEMORY ---
        # This completely eliminates Python list/object overhead.
        # It will take exactly exactly (num_lines * max_len * 2 bytes) of RAM.
        self.src_data = torch.zeros((num_lines, max_s_len), dtype=torch.int16)
        self.trg_data = torch.zeros((num_lines, max_t_len), dtype=torch.int16)
        
        self.src_lens = torch.zeros(num_lines, dtype=torch.int16)
        self.trg_lens = torch.zeros(num_lines, dtype=torch.int16)
        
        max_token_val = sep_token

        # --- PASS 2: Populate the tensors lazily ---
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for idx, (s_line, t_line) in enumerate(zip(f_src, f_trg)):
                src_vals = [int(x) for x in s_line.strip().split()]
                trg_vals = [int(x) for x in t_line.strip().split()]
                
                s_len = len(src_vals)
                t_len = len(trg_vals)
                
                # Fill the pre-allocated row
                self.src_data[idx, :s_len] = torch.tensor(src_vals, dtype=torch.int16)
                self.trg_data[idx, :t_len] = torch.tensor(trg_vals, dtype=torch.int16)
                
                # Record the true lengths so we can slice it accurately later
                self.src_lens[idx] = s_len
                self.trg_lens[idx] = t_len
                
                max_token_val = max(max_token_val, max(src_vals + [0]), max(trg_vals + [0]))

        self.max_token = max_token_val
        self.num_samples = num_lines
        print("Dataset loaded into memory successfully.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve exact lengths for this row
        s_len = self.src_lens[idx].item()
        t_len = self.trg_lens[idx].item()
        
        # Slice out the valid data, ignoring the padding zeros we allocated
        src = self.src_data[idx, :s_len]
        trg = self.trg_data[idx, :t_len]

        sep_tensor = torch.tensor([self.sep_token], dtype=torch.int16)
        seq = torch.cat([src, sep_tensor, trg])

        X = seq[:-1].to(torch.long)

        Y = torch.full((len(seq) - 1,), -100, dtype=torch.long)
        Y[s_len:] = seq[s_len+1:].to(torch.long)

        prompt_len = s_len + 1

        return X, Y, prompt_len

def causal_collate_fn(batch):
    xs, ys, pls = zip(*batch)
    
    X = pad_sequence(xs, batch_first=True, padding_value=0)
    Y = pad_sequence(ys, batch_first=True, padding_value=-100)
    
    return X, Y, torch.tensor(pls, dtype=torch.long)

def validate_em(model, dataloader, config, device):
    model.eval()
    exact_matches = 0
    total_sequences = 0
    
    with torch.no_grad():
        for X, Y, PL in dataloader:
            X, Y = X.to(device), Y.to(device)
            B = X.size(0)
            
            for b in range(B):
                x_seq = X[b]
                y_seq = Y[b]
                prompt_len = PL[b].item()
                
                prompt = x_seq[:prompt_len].unsqueeze(0)
                target_mask = y_seq != -100
                ground_truth = y_seq[target_mask]
                target_len = len(ground_truth)
                
                # --- PREFILL PHASE ---
                with torch.amp.autocast('cuda'):
                    logits, past_kvs = model(prompt, use_cache=True)
                    next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    generated_tokens = [next_token.item()]
                    current_input = next_token
                    
                    # --- DECODE PHASE (FAST) ---
                    for _ in range(1, target_len):
                        logits, past_kvs = model(current_input, use_cache=True, past_kvs=past_kvs)
                        next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                        generated_tokens.append(next_token.item())
                        current_input = next_token

                generated_sequence = torch.tensor(generated_tokens, device=device)
                
                if torch.equal(generated_sequence, ground_truth):
                    exact_matches += 1
                total_sequences += 1

    return exact_matches / total_sequences if total_sequences > 0 else 0.0

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task_dir = os.path.join(args.data_root, f'data_{args.task}')
    train_src = os.path.join(task_dir, 'train.src')
    train_trg = os.path.join(task_dir, 'train.trg')

    train_dataset = Seq2SeqTaskDataset(train_src, train_trg, sep_token=1)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=causal_collate_fn,
        num_workers=0,
        pin_memory=True
    )
    dynamic_vocab_size = train_dataset.max_token + 1

    config = TransformerParams(
        simplex_mapping=SimplexMappingEnum[args.simplex_mapping],
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        int_dim=args.int_dim,
        vocab_size=dynamic_vocab_size, 
        seq_len=256,
        use_nape=True
    ) 

    model = DecoderOnlyTransformer(config).to(device)
    
    # --- MAJOR SPEEDUP: Compile the model to fuse the binary search ops ---
    if hasattr(torch, "compile"):
        print("Compiling model for performance...")
        model = torch.compile(model, dynamic=True)

    is_sparse = (args.simplex_mapping in ["stieltjes", "adaptive_scalable_stieltjes"])
    lr = 5e-4 if is_sparse else 1e-4
    wd = 0.0 if is_sparse else 0.01
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.05,
        anneal_strategy='cos'
    )    

    # --- UPDATED AMP CALLS ---
    scaler = torch.amp.GradScaler('cuda')

    print(f"--- Starting training for task: {args.task} ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (X, Y, _) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(X)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    Y.reshape(-1),
                    ignore_index=-100
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()            
            total_loss += loss.item()
            
            if batch_idx % 1000 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | Train Loss: {loss.item():.4f}")

    print(f"--- Training Complete. Starting OOD Validation --- {args.task} | {args.simplex_mapping}")

    val_files_src = sorted(glob.glob(os.path.join(task_dir, 'validation_*.src')))
    
    effective_batch_size = args.batch_size 
    for i, val_src in enumerate(val_files_src): 
        effective_batch_size = max(2, effective_batch_size // 2)
        val_trg = val_src.replace('.src', '.trg')
        file_name = os.path.basename(val_src).replace('.src', '')
        
        val_dataset = Seq2SeqTaskDataset(val_src, val_trg, sep_token=1)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=effective_batch_size, 
            shuffle=False, 
            collate_fn=causal_collate_fn,
            num_workers=0,
            pin_memory=True
        )        
        em_acc = validate_em(model, val_loader, config, device)
        print(f"[{file_name}] EM Accuracy: {em_acc:.4f}")

        if i > 1: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer on Formal Tasks")
    parser.add_argument("--task", type=str, default="copy", help="Task name (e.g., copy, sort, reverse, mqmtar)")
    parser.add_argument("--simplex_mapping", type=str, default="softmax")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--int_dim", type=int, default=1024)
    parser.add_argument("--data_root", type=str, default="/users/PAS3150/jacktaylor/stieltjes_experiments/asentmax_comp/theory_data/", help="Root directory containing task data folders")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    
    args = parser.parse_args()
    train(args)