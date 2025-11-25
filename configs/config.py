"""
Configuration file for next-location prediction experiments.
"""

import torch


def get_config():
    """Get configuration dictionary."""
    config = {
        # Data settings
        'data_dir': 'data/geolife',
        'max_seq_len': 50,
        'batch_size': 64,
        
        # Model architecture
        'model': {
            'd_model': 128,           # Embedding dimension
            'num_heads': 4,           # Number of attention heads
            'num_layers': 3,          # Number of transformer blocks
            'd_ff': 256,              # Feed-forward hidden dimension
            'dropout': 0.2,           # Dropout rate
        },
        
        # Training settings
        'training': {
            'max_epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1,
            'patience': 15,           # Early stopping patience
            'gradient_clip': 1.0,
        },
        
        # Paths
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'results_dir': 'results',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,
        
        # Random seed
        'seed': 42,
    }
    
    return config
