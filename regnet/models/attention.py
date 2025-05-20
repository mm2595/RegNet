import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SelfAttentionGateFusion(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key   = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.gate  = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores  = torch.matmul(Q, K.T) / (K.size(-1) ** 0.5)
        attn    = F.softmax(scores, dim=-1)                         

        row_entropy = -(attn * (attn.clamp_min(1e-9)).log()).sum(-1).mean()

        attn_out = torch.matmul(attn, V)
        fusion_in = torch.cat([x, attn_out], dim=-1)
        g         = self.gate(fusion_in)
        fused     = g * attn_out + (1 - g) * x
        out       = self.out_proj(fused)
        return out, attn, g, row_entropy      
