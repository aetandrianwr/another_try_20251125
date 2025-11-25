"""
Optimized training script - Version 3
Using parameter-efficient architecture to fit under 500K params.
"""

import os
import random
import numpy as np
import torch
import sys
sys.path.append('/content/another_try_20251125')

from src.data.dataset import load_geolife_data
from src.models.transformer_v3 import NextLocationPredictorV3
from src.utils.trainer import Trainer


def get_config_v3():
    """Optimized configuration for V3 model."""
    config = {
        # Data settings
        'data_dir': 'data/geolife',
        'max_seq_len': 50,
        'batch_size': 32,
        
        # Model architecture - optimized for exactly <500K params
        'model': {
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 3,
            'd_ff': 256,
            'd_loc_emb': 24,  # Factorized location embedding
            'dropout': 0.1,   # Less dropout for smaller model
        },
        
        # Training settings
        'training': {
            'max_epochs': 200,
            'learning_rate': 8e-4,
            'weight_decay': 1e-5,
            'label_smoothing': 0.05,
            'patience': 30,
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    config = get_config_v3()
    set_seed(config['seed'])
    
    print("="*80)
    print("NEXT-LOCATION PREDICTION - VERSION 3 (PARAMETER-EFFICIENT)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {config['device']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Model dimension: {config['model']['d_model']}")
    print(f"  Number of layers: {config['model']['num_layers']}")
    print(f"  Number of heads: {config['model']['num_heads']}")
    print(f"  Feed-forward dim: {config['model']['d_ff']}")
    print(f"  Location embedding dim: {config['model']['d_loc_emb']}")
    print(f"  Dropout: {config['model']['dropout']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("Loading Geolife dataset...")
    print("="*80)
    
    train_loader, val_loader, test_loader, dataset_info = load_geolife_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_len=config['max_seq_len'],
        num_workers=config['num_workers']
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Number of locations: {dataset_info['num_locations']}")
    print(f"  Number of users: {dataset_info['num_users']}")
    print(f"  Train samples: {dataset_info['num_train']}")
    print(f"  Validation samples: {dataset_info['num_val']}")
    print(f"  Test samples: {dataset_info['num_test']}")
    
    # Create model
    print("\n" + "="*80)
    print("Creating parameter-efficient model (V3)...")
    print("="*80)
    
    model = NextLocationPredictorV3(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        d_loc_emb=config['model']['d_loc_emb'],
        max_len=config['max_seq_len'],
        dropout=config['model']['dropout']
    )
    
    num_params = model.count_parameters()
    print(f"\nModel created with {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"❌ WARNING: Model has {num_params:,} parameters (limit: 500,000)")
        raise ValueError("Model exceeds parameter budget!")
    else:
        usage_pct = num_params / 500000 * 100
        print(f"✓ Model is within the 500K parameter budget")
        print(f"  Usage: {usage_pct:.1f}% ({500000 - num_params:,} parameters remaining)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config['device'],
        num_locations=dataset_info['num_locations'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        label_smoothing=config['training']['label_smoothing'],
        max_epochs=config['training']['max_epochs'],
        patience=config['training']['patience'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Train model
    print("\n" + "="*80)
    print("Starting training V3...")
    print("="*80)
    
    test_metrics = trainer.train()
    
    # Save results
    results_path = os.path.join(config['results_dir'], 'final_results_v3.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL TEST RESULTS - VERSION 3 (PARAMETER-EFFICIENT)\n")
        f.write("="*80 + "\n")
        f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n")
        f.write(f"\nModel Parameters: {num_params:,} / 500,000\n")
        f.write(f"Parameter Usage: {num_params/500000*100:.1f}%\n")
        f.write(f"Best Validation Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Best Epoch: {trainer.best_epoch + 1}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("ARCHITECTURE HIGHLIGHTS:\n")
        f.write(f"{'='*80}\n")
        f.write("- Factorized location embeddings (24-dim bottleneck)\n")
        f.write("- 3-layer transformer with 128-dim hidden states\n")
        f.write("- 4 attention heads for multi-perspective modeling\n")
        f.write("- Learnable positional embeddings\n")
        f.write("- Pre-layer normalization for training stability\n")
        f.write("- GELU activations\n")
    
    print(f"\nResults saved to {results_path}")
    
    return test_metrics


if __name__ == '__main__':
    main()
