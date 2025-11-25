# Next-Location Prediction on Geolife Dataset

A state-of-the-art PyTorch implementation for next-location prediction using transformer-based architecture inspired by Sequential Recommendation Systems.

## 🎯 Objective

Achieve **≥40% Test Acc@1** on the Geolife dataset with **<500K parameters**.

## 🏗️ Architecture

The model draws from cutting-edge Sequential Recommendation research:

- **SASRec**: Self-attention mechanism for capturing sequential patterns
- **BERT4Rec**: Bidirectional context understanding
- **Transformer blocks**: Multi-head self-attention with position-wise feed-forward networks
- **Rich features**: Location, user, temporal (weekday, time, duration) embeddings

### Key Design Decisions

1. **Self-Attention over RNNs**: Better at capturing long-range dependencies and parallel processing
2. **Positional Encoding**: Sinusoidal encoding preserves sequence order information
3. **Label Smoothing**: Prevents overconfidence, improves generalization (0.1 smoothing)
4. **Cosine Annealing with Warm Restarts**: Dynamic learning rate for better convergence
5. **Multi-scale Features**: Combines categorical embeddings with continuous temporal features
6. **Causal Masking**: Prevents information leakage from future positions

## 📊 Model Specifications

- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 3
- **Feed-Forward Dimension**: 256
- **Dropout**: 0.2
- **Total Parameters**: ~450K (within 500K budget)

## 🚀 Training Strategy

- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-3 with cosine annealing
- **Batch Size**: 64
- **Max Epochs**: 100
- **Early Stopping**: Patience of 15 epochs
- **Gradient Clipping**: Max norm of 1.0

## 📁 Project Structure

```
.
├── configs/
│   └── config.py              # Hyperparameters and settings
├── src/
│   ├── data/
│   │   └── dataset.py         # Data loading and preprocessing
│   ├── models/
│   │   └── transformer_model.py  # Model architecture
│   └── utils/
│       ├── metrics.py         # Evaluation metrics
│       └── trainer.py         # Training loop
├── train.py                   # Main training script
├── data/geolife/              # Dataset directory
├── checkpoints/               # Saved models
├── logs/                      # Training logs
└── results/                   # Final results
```

## 🔧 Usage

```bash
# Train the model
python3 train.py
```

## 📈 Expected Performance

- **Validation Acc@1**: ~38-42%
- **Test Acc@1**: ≥40% (target)
- **Test Acc@5**: ~60-65%
- **MRR**: ~45-50%

## 🧠 Research Insights

### Why This Architecture?

1. **Sequential Recommendation → Trajectory Prediction**: The problem structure is identical - predicting the next item given a sequence of past items
2. **Attention Mechanisms**: Superior to RNNs for:
   - Long-range dependencies
   - Parallel training
   - Interpretability
3. **Temporal Features**: Time-of-day and duration patterns are crucial for location prediction
4. **User Context**: User-specific preferences significantly impact location choices

### Regularization Strategy

- **Label Smoothing**: Prevents overfitting to training data
- **Dropout**: Applied to attention weights and feed-forward layers
- **Weight Decay**: L2 regularization on all parameters
- **Early Stopping**: Prevents training beyond optimal point

## 📚 References

- Kang & McAuley (2018). "Self-Attentive Sequential Recommendation" (SASRec)
- Sun et al. (2019). "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
- Vaswani et al. (2017). "Attention Is All You Need"

## 📊 Dataset Statistics

- **Train Samples**: 7,424
- **Validation Samples**: 3,334
- **Test Samples**: 3,502
- **Unique Locations**: 1,187
- **Unique Users**: 45
- **Avg Sequence Length**: 18

---

**Author**: Research Implementation for Geolife Next-Location Prediction  
**Target**: 40% Test Acc@1 with <500K parameters
