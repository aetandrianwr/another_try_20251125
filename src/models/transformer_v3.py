"""
Parameter-Efficient Next-Location Prediction Model - Version 3

Key strategy: Use factorized embeddings to reduce parameter count
while maintaining model capacity in transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FactorizedEmbedding(nn.Module):
    """
    Factorized embedding to reduce parameters.
    Instead of (vocab_size, d_model), use (vocab_size, d_emb) -> (d_emb, d_model)
    """
    def __init__(self, vocab_size, d_model, d_emb=64, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=padding_idx)
        self.projection = nn.Linear(d_emb, d_model, bias=False)
        
    def forward(self, x):
        return self.projection(self.embedding(x))


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.positional_embedding(positions)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
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
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Padding mask
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x


class NextLocationPredictorV3(nn.Module):
    """
    Parameter-efficient transformer for next-location prediction.
    
    Key optimizations:
    - Factorized location embeddings (major savings)
    - Shared embedding for output projection
    - Efficient feature fusion
    - Optimized for <500K parameters
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=160,
        num_heads=4,
        num_layers=4,
        d_ff=320,
        max_len=50,
        dropout=0.15,
        d_loc_emb=96,  # Factorized location embedding dimension
        num_weekdays=7,
        max_time_diff=100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Factorized location embedding (saves parameters!)
        self.location_embedding = FactorizedEmbedding(
            num_locations, d_model, d_emb=d_loc_emb, padding_idx=0
        )
        
        # User embedding (smaller)
        self.user_embedding = nn.Embedding(num_users, d_model // 8, padding_idx=0)
        
        # Temporal embeddings (compact)
        self.weekday_embedding = nn.Embedding(num_weekdays + 1, d_model // 16, padding_idx=0)
        self.time_diff_embedding = nn.Embedding(max_time_diff + 1, d_model // 16, padding_idx=0)
        
        # Continuous feature projections (very small)
        self.duration_projection = nn.Linear(1, d_model // 16)
        self.start_min_projection = nn.Linear(1, d_model // 16)
        
        # Feature fusion
        feature_dim = d_model + d_model // 8 + 4 * (d_model // 16)
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection - weight sharing with location embedding for efficiency
        self.output_hidden = nn.Linear(d_model, d_loc_emb)
        self.output_layer = nn.Linear(d_loc_emb, num_locations)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                if 'weight' in name:
                    nn.init.normal_(param, mean=0, std=0.02)
            elif any(key in name for key in ['linear', 'projection']):
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
        
        # Clip values
        time_diffs = torch.clamp(time_diffs, 0, 100)
        
        # Get embeddings
        loc_emb = self.location_embedding(locations)
        user_emb = self.user_embedding(users)
        weekday_emb = self.weekday_embedding(weekdays)
        time_diff_emb = self.time_diff_embedding(time_diffs)
        
        # Project continuous features
        dur_emb = self.duration_projection(durations.unsqueeze(-1))
        start_min_emb = self.start_min_projection(start_mins.unsqueeze(-1).float())
        
        # Concatenate and fuse features
        combined = torch.cat([
            loc_emb, user_emb, weekday_emb, time_diff_emb, dur_emb, start_min_emb
        ], dim=-1)
        
        x = self.feature_fusion(combined)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.final_norm(x)
        
        # Extract last position
        last_positions = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden = x[batch_indices, last_positions]
        
        # Output projection
        hidden = F.gelu(self.output_hidden(last_hidden))
        hidden = self.dropout(hidden)
        logits = self.output_layer(hidden)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
