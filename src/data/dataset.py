"""
Geolife Dataset Loader with PyTorch Dataset/DataLoader abstractions.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeolifeDataset(Dataset):
    """
    PyTorch Dataset for Geolife trajectory data.
    
    Each sample contains:
        - X: location sequence (trajectory history)
        - user_X: user IDs for each location
        - weekday_X: day of week for each visit
        - start_min_X: start time in minutes
        - dur_X: duration at each location
        - diff: time difference features
        - Y: target next location
    """
    
    def __init__(self, data, max_len=50):
        """
        Args:
            data: List of dictionaries from pickle file
            max_len: Maximum sequence length for padding/truncating
        """
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features
        locations = sample['X']  # location IDs
        users = sample['user_X']
        weekdays = sample['weekday_X']
        start_mins = sample['start_min_X']
        durations = sample['dur_X']
        time_diffs = sample['diff']
        target = sample['Y']
        
        seq_len = len(locations)
        
        # Pad or truncate to max_len
        if seq_len > self.max_len:
            # Take most recent locations
            locations = locations[-self.max_len:]
            users = users[-self.max_len:]
            weekdays = weekdays[-self.max_len:]
            start_mins = start_mins[-self.max_len:]
            durations = durations[-self.max_len:]
            time_diffs = time_diffs[-self.max_len:]
            seq_len = self.max_len
        
        # Create padding mask (1 for real tokens, 0 for padding)
        mask = np.ones(self.max_len, dtype=np.float32)
        
        if seq_len < self.max_len:
            pad_len = self.max_len - seq_len
            # Pad with zeros
            locations = np.concatenate([np.zeros(pad_len, dtype=locations.dtype), locations])
            users = np.concatenate([np.zeros(pad_len, dtype=users.dtype), users])
            weekdays = np.concatenate([np.zeros(pad_len, dtype=weekdays.dtype), weekdays])
            start_mins = np.concatenate([np.zeros(pad_len, dtype=start_mins.dtype), start_mins])
            durations = np.concatenate([np.zeros(pad_len, dtype=durations.dtype), durations])
            time_diffs = np.concatenate([np.zeros(pad_len, dtype=time_diffs.dtype), time_diffs])
            mask[:pad_len] = 0
        
        return {
            'locations': torch.LongTensor(locations),
            'users': torch.LongTensor(users),
            'weekdays': torch.LongTensor(weekdays),
            'start_mins': torch.LongTensor(start_mins),
            'durations': torch.FloatTensor(durations),
            'time_diffs': torch.LongTensor(time_diffs),
            'mask': torch.FloatTensor(mask),
            'target': torch.LongTensor([target]),
            'seq_len': seq_len
        }


def load_geolife_data(data_dir, batch_size=128, max_len=50, num_workers=0):
    """
    Load Geolife datasets and create DataLoaders.
    
    Args:
        data_dir: Directory containing pickle files
        batch_size: Batch size for training
        max_len: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Load data
    with open(f'{data_dir}/geolife_transformer_7_train.pk', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f'{data_dir}/geolife_transformer_7_validation.pk', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(f'{data_dir}/geolife_transformer_7_test.pk', 'rb') as f:
        test_data = pickle.load(f)
    
    # Gather dataset statistics
    all_locs = set()
    all_users = set()
    for sample in train_data + val_data + test_data:
        all_locs.update(sample['X'])
        all_users.update(sample['user_X'])
    
    dataset_info = {
        'num_locations': max(all_locs) + 1,  # +1 for padding token at 0
        'num_users': max(all_users) + 1,
        'num_train': len(train_data),
        'num_val': len(val_data),
        'num_test': len(test_data),
    }
    
    # Create datasets
    train_dataset = GeolifeDataset(train_data, max_len=max_len)
    val_dataset = GeolifeDataset(val_data, max_len=max_len)
    test_dataset = GeolifeDataset(test_data, max_len=max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset_info
