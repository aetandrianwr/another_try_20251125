"""
Training utilities and helper functions.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for better generalization.
    
    Prevents the model from becoming overconfident and improves calibration.
    Reference: "Rethinking the Inception Architecture for Computer Vision"
    """
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes) raw scores
            targets: (B,) ground truth class indices
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # One-hot encode targets with smoothing
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class Trainer:
    """Training and evaluation engine."""
    
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        device,
        num_locations,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.1,
        max_epochs=100,
        patience=15,
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(num_locations, smoothing=label_smoothing)
        
        # Optimizer: AdamW with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.98)  # Following BERT/Transformer conventions
        )
        
        # Learning rate scheduler: cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # Initial restart period
            T_mult=2,  # Multiply period by 2 after each restart
            eta_min=1e-6
        )
        
        # Track best model
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            targets = batch['target'].squeeze(-1)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            metrics, _, _ = calculate_correct_total_prediction(logits.detach(), targets)
            all_metrics.append(metrics)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        metrics_dict = self._aggregate_metrics(all_metrics)
        
        return avg_loss, metrics_dict
    
    def evaluate(self, data_loader, desc='Validation'):
        """Evaluate on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=desc, leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                targets = batch['target'].squeeze(-1)
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                metrics, _, _ = calculate_correct_total_prediction(logits, targets)
                all_metrics.append(metrics)
        
        avg_loss = total_loss / len(data_loader)
        metrics_dict = self._aggregate_metrics(all_metrics)
        
        return avg_loss, metrics_dict
    
    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics across batches."""
        # Sum all metrics
        total_metrics = np.sum(metrics_list, axis=0)
        
        return_dict = {
            "correct@1": total_metrics[0],
            "correct@3": total_metrics[1],
            "correct@5": total_metrics[2],
            "correct@10": total_metrics[3],
            "rr": total_metrics[4],
            "ndcg": total_metrics[5],
            "f1": 0.0,  # F1 not computed in batch mode
            "total": total_metrics[6],
        }
        
        return get_performance_dict(return_dict)
    
    def save_checkpoint(self, epoch, val_acc, filepath):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def train(self):
        """Full training loop with early stopping."""
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(self.max_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.evaluate(self.val_loader, desc='Validation')
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_metrics['acc@1'])
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['acc@1'])
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc@1: {train_metrics['acc@1']:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_metrics['acc@1']:.2f}%")
            print(f"Val Acc@5: {val_metrics['acc@5']:.2f}% | Val MRR: {val_metrics['mrr']:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Check for improvement
            if val_metrics['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc@1']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                checkpoint_path = f"{self.checkpoint_dir}/best_model.pt"
                self.save_checkpoint(epoch, val_metrics['acc@1'], checkpoint_path)
                print(f"✓ New best model saved! Val Acc@1: {val_metrics['acc@1']:.2f}%")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
                break
        
        # Load best model and evaluate on test set
        print(f"\n{'='*60}")
        print("Loading best model and evaluating on test set...")
        print(f"{'='*60}")
        
        checkpoint_path = f"{self.checkpoint_dir}/best_model.pt"
        self.load_checkpoint(checkpoint_path)
        
        test_loss, test_metrics = self.evaluate(self.test_loader, desc='Test')
        
        print(f"\n{'='*60}")
        print("FINAL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc@1: {test_metrics['acc@1']:.2f}%")
        print(f"Test Acc@5: {test_metrics['acc@5']:.2f}%")
        print(f"Test Acc@10: {test_metrics['acc@10']:.2f}%")
        print(f"Test MRR: {test_metrics['mrr']:.2f}%")
        print(f"Test NDCG: {test_metrics['ndcg']:.2f}%")
        print(f"{'='*60}")
        
        return test_metrics
