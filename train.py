"""
Main training script for next-location prediction on Geolife dataset.

This script implements a state-of-the-art sequential recommendation approach
adapted for trajectory prediction, drawing from:
- SASRec (Self-Attentive Sequential Recommendation)
- BERT4Rec (Bidirectional transformers for sequential recommendation)
- Modern transformer architectures with careful regularization

Key innovations:
1. Self-attention mechanism for long-range dependencies
2. Rich temporal and user context features
3. Label smoothing for better generalization
4. Cosine annealing with warm restarts
5. Early stopping based on validation performance
"""

import os
import random
import numpy as np
import torch
import sys
sys.path.append('/content/another_try_20251125')

from src.data.dataset import load_geolife_data
from src.models.transformer_model import NextLocationPredictor
from src.utils.trainer import Trainer
from configs.config import get_config


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
    # Load configuration
    config = get_config()
    
    # Set random seed
    set_seed(config['seed'])
    
    print("="*80)
    print("NEXT-LOCATION PREDICTION ON GEOLIFE DATASET")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {config['device']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Max sequence length: {config['max_seq_len']}")
    print(f"  Model dimension: {config['model']['d_model']}")
    print(f"  Number of layers: {config['model']['num_layers']}")
    print(f"  Number of heads: {config['model']['num_heads']}")
    print(f"  Dropout: {config['model']['dropout']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Label smoothing: {config['training']['label_smoothing']}")
    
    # Create directories
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
    print("Creating model...")
    print("="*80)
    
    model = NextLocationPredictor(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_len=config['max_seq_len'],
        dropout=config['model']['dropout']
    )
    
    num_params = model.count_parameters()
    print(f"\nModel created with {num_params:,} parameters")
    
    if num_params >= 500000:
        print(f"WARNING: Model has {num_params:,} parameters (limit: 500,000)")
    else:
        print(f"✓ Model is within the 500K parameter budget ({500000 - num_params:,} parameters remaining)")
    
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
    print("Starting training...")
    print("="*80)
    
    test_metrics = trainer.train()
    
    # Save final results
    results_path = os.path.join(config['results_dir'], 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n")
        f.write(f"\nModel Parameters: {num_params:,}\n")
        f.write(f"Best Validation Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Best Epoch: {trainer.best_epoch + 1}\n")
    
    print(f"\nResults saved to {results_path}")
    
    return test_metrics


if __name__ == '__main__':
    main()
