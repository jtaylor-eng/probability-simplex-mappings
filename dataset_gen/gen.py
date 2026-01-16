import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


def make_dataset(
    len_dataset: int,
    max_len_seq: int,
    n_classes: int = 10,
    min_len_seq: int = 5,
    batch_size: int = 128,
    shuffle: bool = True,
):
    """
    Creates a dataset for the max retrieval problem.
    
    Each sample consists of:
    - items: (T, 1 + n_classes) tensor with priorities and one-hot encoded classes
    - query_vec: (1,) tensor with a random query value
    - target_class: the class of the item with maximum priority
    
    Args:
        len_dataset: Number of samples in the dataset
        max_len_seq: Maximum sequence length
        n_classes: Number of classes
        min_len_seq: Minimum sequence length
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset
        
    Returns:
        DataLoader for the max retrieval dataset
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
        items = torch.cat([priorities_t, classes_t], dim=-1)  # (T, 1 + n_classes)
        
        query_vec = torch.tensor([np.random.uniform(0, 1)]).float()
        
        data.append((items, query_vec, torch.tensor(target_class).long()))

    dataset = MaxDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_max_data,
        num_workers=4
    )

