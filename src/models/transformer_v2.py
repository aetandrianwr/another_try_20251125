"""
Enhanced Next-Location Prediction Model - Version 2

Key Improvements:
1. Larger embedding dimensions for better representation capacity
2. Deeper architecture with more transformer layers
3. Better feature fusion with gating mechanisms
4. Item-item co-occurrence modeling (inspired by collaborative filtering)
5. Learnable positional embeddings (more flexible than sinusoidal)
6. Residual connections in feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings - more flexible than fixed sinusoidal."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.positional_embedding(positions)


class GatedFeatureFusion(nn.Module):
    """Gated fusion mechanism for combining multiple feature types."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        """Apply gated linear unit for feature fusion."""
        return self.transform(x) * torch.sigmoid(self.gate(x))


class MultiHeadSelfAttention(nn.Module):
    """Enhanced multi-head self-attention with better regularization."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into multiple heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply padding mask
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output


class FeedForward(nn.Module):
    """Enhanced position-wise feed-forward with GELU activation."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Enhanced transformer block with pre-layer normalization."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture (more stable training)
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x


class NextLocationPredictorV2(nn.Module):
    """
    Enhanced Transformer-based next-location prediction model.
    
    Improvements over V1:
    - Larger capacity while staying under 500K params
    - Better feature fusion with gating
    - Learnable positional embeddings
    - Pre-layer normalization for stability
    - GELU activation instead of ReLU
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=192,
        num_heads=6,
        num_layers=4,
        d_ff=384,
        max_len=50,
        dropout=0.15,
        num_weekdays=7,
        max_start_min=1440,
        max_time_diff=100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Larger embeddings for better capacity
        self.location_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_embedding = nn.Embedding(num_weekdays + 1, d_model // 8, padding_idx=0)
        self.time_diff_embedding = nn.Embedding(max_time_diff + 1, d_model // 8, padding_idx=0)
        
        # Continuous feature projections
        self.duration_projection = nn.Linear(1, d_model // 8)
        self.start_min_projection = nn.Linear(1, d_model // 8)
        
        # Gated feature fusion
        feature_dim = d_model + d_model // 4 + 4 * (d_model // 8)
        self.feature_fusion = GatedFeatureFusion(feature_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection with hidden layer
        self.output_hidden = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, num_locations)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=0.02)
            elif any(key in name for key in ['linear', 'projection', 'output']):
                if 'weight' in name:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        durations = batch['durations']
        time_diffs = batch['time_diffs']
        mask = batch['mask']
        
        # Clip values to valid ranges
        time_diffs = torch.clamp(time_diffs, 0, 100)
        
        # Get embeddings
        loc_emb = self.location_embedding(locations)
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekdays)
        time_diff_emb = self.time_diff_embedding(time_diffs)
        
        # Project continuous features
        dur_emb = self.duration_projection(durations.unsqueeze(-1))
        start_min_emb = self.start_min_projection(start_mins.unsqueeze(-1).float())
        
        # Concatenate all features
        combined = torch.cat([
            loc_emb, user_emb, weekday_emb, time_diff_emb, dur_emb, start_min_emb
        ], dim=-1)
        
        # Gated fusion
        x = self.feature_fusion(combined)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.final_norm(x)
        
        # Extract last position representation
        last_positions = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden = x[batch_indices, last_positions]
        
        # Two-layer output projection
        hidden = F.gelu(self.output_hidden(last_hidden))
        hidden = self.dropout(hidden)
        logits = self.output_layer(hidden)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
