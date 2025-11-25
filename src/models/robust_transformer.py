"""
Final Robust Model - Combining proven techniques

After extensive testing, this model uses only battle-tested components:
- Standard embedding layers
- Multi-head attention (PyTorch native - stable)
- Proper initialization
- Gradient clipping
- Label smoothing

Target: Reliable 40%+ Test Acc@1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RobustTransformer(nn.Module):
    """
    Rock-solid transformer implementation.
    Uses PyTorch's native components for maximum stability.
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=112,
        nhead=8,
        num_layers=3,
        dim_feedforward=224,
        dropout=0.1,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Auxiliary embeddings (smaller)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.weekday_emb = nn.Embedding(8, d_model // 8, padding_idx=0)
        self.hour_emb = nn.Embedding(25, d_model // 8, padding_idx=0)
        
        # Feature projections
        self.feat_proj = nn.Sequential(
            nn.Linear(d_model + d_model // 4 + 2 * (d_model // 8), d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder (native PyTorch - very stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Careful initialization to avoid NaNs."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)  # Smaller gain for stability
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate causal mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        B, L = locations.shape
        device = locations.device
        
        # Get hours
        hours = torch.clamp(start_mins // 60, 0, 24)
        
        # Embeddings
        loc = self.loc_emb(locations)
        user = self.user_emb(users)
        weekday = self.weekday_emb(weekdays)
        hour = self.hour_emb(hours)
        
        # Position
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos)
        
        # Combine features
        combined = torch.cat([loc, user, weekday, hour], dim=-1)
        x = self.feat_proj(combined) + pos_emb
        
        # Masks
        causal_mask = self.generate_square_subsequent_mask(L, device)
        src_key_padding_mask = ~mask.bool()
        
        # Transform
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Get last valid position
        lengths = mask.sum(dim=1).long() - 1
        idx = torch.arange(B, device=device)
        last_hidden = x[idx, lengths]
        
        # Output
        logits = self.output(last_hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
