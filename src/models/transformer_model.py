"""
Advanced Next-Location Prediction Model.

Architecture inspired by state-of-the-art Sequential Recommendation Systems:
- SASRec (Self-Attentive Sequential Recommendation)
- BERT4Rec (Bidirectional Encoder Representations from Transformers)
- GRU4Rec concepts adapted to Transformer architecture

Key Design Principles:
1. Self-attention mechanisms for capturing long-range dependencies
2. Positional encoding for sequence order
3. Rich temporal and user context features
4. Careful regularization to prevent overfitting
5. Efficient architecture (<500K parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention is All You Need'."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking for autoregressive prediction."""
    
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
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - padding mask
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and split into multiple heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (prevent attending to future positions)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply padding mask
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(padding_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class NextLocationPredictor(nn.Module):
    """
    Transformer-based next-location prediction model.
    
    Inspired by SASRec with additional temporal and user features.
    Designed to stay under 500K parameters while achieving high accuracy.
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        num_heads=4,
        num_layers=3,
        d_ff=256,
        max_len=50,
        dropout=0.2,
        num_weekdays=7,
        max_start_min=1440,
        max_time_diff=100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Embedding layers
        self.location_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Temporal embeddings (smaller dimension to save parameters)
        self.weekday_embedding = nn.Embedding(num_weekdays + 1, d_model // 8, padding_idx=0)
        self.time_diff_embedding = nn.Embedding(max_time_diff + 1, d_model // 8, padding_idx=0)
        
        # Continuous feature projection
        self.duration_projection = nn.Linear(1, d_model // 8)
        self.start_min_projection = nn.Linear(1, d_model // 8)
        
        # Feature fusion
        feature_dim = d_model + d_model // 4 + 4 * (d_model // 8)
        self.feature_fusion = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, num_locations)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'linear' in name or 'projection' in name or 'output_layer' in name:
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                - locations: (B, L) location sequence
                - users: (B, L) user IDs
                - weekdays: (B, L) day of week
                - start_mins: (B, L) start time in minutes
                - durations: (B, L) duration at location
                - time_diffs: (B, L) time difference
                - mask: (B, L) padding mask
        
        Returns:
            logits: (B, num_locations) prediction scores
        """
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
        loc_emb = self.location_embedding(locations)  # (B, L, d_model)
        user_emb = self.user_embedding(users)  # (B, L, d_model//4)
        weekday_emb = self.weekday_embedding(weekdays)  # (B, L, d_model//8)
        time_diff_emb = self.time_diff_embedding(time_diffs)  # (B, L, d_model//8)
        
        # Project continuous features
        dur_emb = self.duration_projection(durations.unsqueeze(-1))  # (B, L, d_model//8)
        start_min_emb = self.start_min_projection(start_mins.unsqueeze(-1).float())  # (B, L, d_model//8)
        
        # Concatenate all features
        combined = torch.cat([
            loc_emb, user_emb, weekday_emb, time_diff_emb, dur_emb, start_min_emb
        ], dim=-1)
        
        # Fuse features to d_model dimension
        x = self.feature_fusion(combined)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Use the last position for prediction (autoregressive)
        # Take the representation of the last real token for each sequence
        last_positions = mask.sum(dim=1).long() - 1  # (B,)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden = x[batch_indices, last_positions]  # (B, d_model)
        
        # Project to location space
        logits = self.output_layer(last_hidden)  # (B, num_locations)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
