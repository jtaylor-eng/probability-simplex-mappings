import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle
from pathlib import Path
import argparse


class MaxDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_max_data(batch):
    """Collates (items, query_vec, target_class)"""
    items, queries, targets = zip(*batch) 
    
    items_padded = rnn_utils.pad_sequence(items, batch_first=True, padding_value=0.0)
    queries_stacked = torch.stack(queries)
    targets_stacked = torch.stack(targets)
    
    return items_padded, queries_stacked, targets_stacked


def _generate_data(
    len_dataset: int,
    max_len_seq: int,
    n_classes: int = 10,
    min_len_seq: int = 5,
):
    """
    Generates raw data for the max retrieval problem.
    
    Returns:
        List of tuples: (items, query_vec, target_class)
    """
    data = []
    for _ in tqdm(range(len_dataset), desc='creating dataset'):
        current_len = np.random.randint(min_len_seq, max_len_seq + 1)
            
        priorities = np.random.uniform(0, 1, size=current_len)
        classes = np.random.randint(0, n_classes, size=current_len)
        
        target_class_idx = np.argmax(priorities)
        target_class = classes[target_class_idx]
        
        priorities_t = torch.tensor(priorities).float().unsqueeze(-1)
        classes_t = F.one_hot(torch.tensor(classes), n_classes).float()
        items = torch.cat([priorities_t, classes_t], dim=-1)  # (T, 1 + args.n_classes)
        
        query_vec = torch.tensor([np.random.uniform(0, 1)]).float()
        
        data.append((items, query_vec, torch.tensor(target_class).long()))
    
    return data


def _get_dataset_filename(len_dataset: int, max_len_seq: int, n_classes: int, min_len_seq: int):
    """Generate a filename for a dataset based on its parameters."""
    return f"dataset_{len_dataset}_{max_len_seq}_{n_classes}_{min_len_seq}.pkl"


def save_dataset(
    data: list,
    len_dataset: int,
    max_len_seq: int,
    n_classes: int,
    min_len_seq: int,
    data_dir: str = "simplex_mappings/data"
):
    """Save a dataset to disk."""
    os.makedirs(data_dir, exist_ok=True)
    filename = _get_dataset_filename(len_dataset, max_len_seq, n_classes, min_len_seq)
    filepath = os.path.join(data_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved dataset to {filepath}")
    return filepath


def load_dataset(
    len_dataset: int,
    max_len_seq: int,
    n_classes: int,
    min_len_seq: int,
    data_dir: str = "simplex_mappings/data"
):
    """Load a dataset from disk."""
    filename = _get_dataset_filename(len_dataset, max_len_seq, n_classes, min_len_seq)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded dataset from {filepath}")
    return data


def make_dataset(
    len_dataset: int,
    max_len_seq: int,
    n_classes: int = 10,
    min_len_seq: int = 5,
    batch_size: int = 128,
    shuffle: bool = True,
    data_dir: str = "simplex_mappings/data",
    force_regenerate: bool = False,
):
    """
    Creates a dataset for the max retrieval problem.
    First tries to load from disk, otherwise generates and saves it.
    
    Each sample consists of:
    - items: (T, 1 + args.n_classes) tensor with priorities and one-hot encoded classes
    - query_vec: (1,) tensor with a random query value
    - target_class: the class of the item with maximum priority
    
    Args:
        len_dataset: Number of samples in the dataset
        max_len_seq: Maximum sequence length
        args.n_classes: Number of classes
        args.min_len_seq: Minimum sequence length
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        data_dir: Directory to save/load datasets from
        force_regenerate: If True, regenerate even if file exists
        
    Returns:
        DataLoader for the max retrieval dataset
    """
    #load from disk first if exists
    if not force_regenerate:
        data = load_dataset(len_dataset, max_len_seq, n_classes, min_len_seq, data_dir)
        if data is not None:
            dataset = MaxDataset(data)
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                collate_fn=collate_max_data,
                num_workers=4
            )
    
    data = _generate_data(len_dataset, max_len_seq, n_classes, min_len_seq)
    save_dataset(data, len_dataset, max_len_seq, n_classes, min_len_seq, data_dir)
    
    dataset = MaxDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_max_data,
        num_workers=4
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=10, help="The name of the user to greet.")
    parser.add_argument("--min_len_seq", type=int, default=5, help="The name of the user to greet.")
    parser.add_argument("--max_len_seq", type=int, default=16, help="The name of the user to greet.")
    parser.add_argument("--len_train_dataset", type=int, default=20000*64, help="The name of the user to greet.")

    args = parser.parse_args()

    data_dir = str(Path(__file__).parent.parent / "data")

    #train dataset
    train_data = _generate_data(args.len_train_dataset, args.max_len_seq, args.n_classes, args.min_len_seq)
    save_dataset(
        train_data,
        args.len_train_dataset, args.max_len_seq, args.n_classes, args.min_len_seq, data_dir
    )
    
    #val ID dataset
    len_val_dataset = 1024
    max_len_seq_val_ID = 16
    print(f"Generating validation ID dataset: {len_val_dataset} samples, max_len={max_len_seq_val_ID}")
    val_ID_data = _generate_data(len_val_dataset, max_len_seq_val_ID, args.n_classes, args.min_len_seq)
    save_dataset(
        val_ID_data,
        len_val_dataset, max_len_seq_val_ID, args.n_classes, args.min_len_seq, data_dir
    )
    
    #val OOD dataset
    max_len_seq_val_OOD = 128
    print(f"Generating validation OOD dataset: {len_val_dataset} samples, max_len={max_len_seq_val_OOD}")
    val_OOD_data = _generate_data(len_val_dataset, max_len_seq_val_OOD, args.n_classes, args.min_len_seq)
    save_dataset(
        val_OOD_data,
        len_val_dataset, max_len_seq_val_OOD, args.n_classes, args.min_len_seq, data_dir
    )