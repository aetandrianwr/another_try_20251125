"""
Simple but Effective Transformer Model - Version 5

Philosophy: Keep it simple, proven, and focused.

Based on proven techniques:
1. Standard transformer with proper normalization
2. Rich input features  
3. Good regularization
4. Proper training techniques

No fancy stuff - just solid engineering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleTransformerBlock(nn.Module):
    """Standard transformer block - nothing fancy."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-norm transformer
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        x = x + self.ff(self.norm2(x))
        
        return x


class SimpleTransformerModel(nn.Module):
    """
    Simple, reliable transformer for next-location prediction.
    
    Design principles:
    - Use proven transformer architecture
    - Rich input features
    - Moderate capacity
    - Stay under 500K params
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        num_heads=8,
        num_layers=3,
        d_ff=256,
        max_len=50,
        dropout=0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_locations = num_locations
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Temporal embeddings
        self.weekday_emb = nn.Embedding(8, d_model // 8, padding_idx=0)  # 0-6 + padding
        self.hour_emb = nn.Embedding(25, d_model // 8, padding_idx=0)  # 0-24
        
        # Feature projections
        self.duration_proj = nn.Linear(1, d_model // 16)
        self.time_diff_proj = nn.Linear(1, d_model // 16)
        
        # Input projection
        input_dim = d_model + d_model // 4 + 2 * (d_model // 8) + 2 * (d_model // 16)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def create_causal_mask(self, seq_len, device):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        durations = batch['durations']
        time_diffs = batch['time_diffs']
        mask = batch['mask']
        
        B, L = locations.shape
        device = locations.device
        
        # Extract hours from start_mins
        hours = torch.clamp(start_mins // 60, 0, 24)
        
        # Normalize continuous features
        durations_norm = (durations / 60.0).unsqueeze(-1)  # Normalize to hours
        time_diffs_norm = torch.clamp(time_diffs / 10.0, 0, 10).unsqueeze(-1).float()  # Normalize
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        weekday_emb = self.weekday_emb(weekdays)
        hour_emb = self.hour_emb(hours)
        dur_emb = self.duration_proj(durations_norm)
        td_emb = self.time_diff_proj(time_diffs_norm)
        
        # Position embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(positions)
        
        # Combine features
        combined = torch.cat([
            loc_emb, user_emb, weekday_emb, hour_emb, dur_emb, td_emb
        ], dim=-1)
        
        # Project to model dimension and add positional encoding
        x = self.input_proj(combined) + pos_emb
        
        # Create masks
        causal_mask = self.create_causal_mask(L, device)
        key_padding_mask = ~mask.bool()  # True for padding positions
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        x = self.norm(x)
        
        # Get last valid position for each sequence
        last_positions = mask.sum(dim=1).long() - 1
        batch_indices = torch.arange(B, device=device)
        last_hidden = x[batch_indices, last_positions]
        
        # Output projection
        logits = self.output(last_hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
