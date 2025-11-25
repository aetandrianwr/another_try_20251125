"""
Ultra-Stable Training Script

Maximum safeguards against NaN and instability.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('/content/another_try_20251125')

from src.data.dataset import load_geolife_data
from src.models.robust_transformer import RobustTransformer
from src.utils.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    config = {
        'data_dir': 'data/geolife',
        'max_seq_len': 50,
        'batch_size': 64,
        'seed': 42,
        
        'model': {
            'd_model': 96,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 192,
            'dropout': 0.1,
        },
        
        'training': {
            'max_epochs': 200,
            'learning_rate': 5e-4,  # Conservative LR
            'weight_decay': 1e-4,
            'label_smoothing': 0.1,
            'patience': 30,
        },
        
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    set_seed(config['seed'])
    
    print("="*80)
    print("ROBUST TRANSFORMER - STABLE TRAINING")
    print("="*80)
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_info = load_geolife_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_len=config['max_seq_len'],
        num_workers=0
    )
    
    print(f"Dataset: {dataset_info['num_locations']} locations, {dataset_info['num_users']} users")
    
    # Create model
    print("\nCreating model...")
    model = RobustTransformer(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        **config['model'],
        max_len=config['max_seq_len']
    )
    
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} ({num_params/500000*100:.1f}% of budget)")
    
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
    
    # Train
    print("\nStarting training...")
    test_metrics = trainer.train()
    
    # Save results
    results_path = os.path.join(config['results_dir'], 'final_results_robust.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROBUST TRANSFORMER - FINAL RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n\n")
        f.write(f"Model Parameters: {num_params:,}\n")
        f.write(f"Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
    
    print(f"\nResults saved to {results_path}")
    
    return test_metrics


if __name__ == '__main__':
    main()
