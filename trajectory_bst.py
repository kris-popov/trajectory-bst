"""
Trajectory Belief State Transformer:

Implementation based on, https://arxiv.org/pdf/2410.23506


"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT_Backbone

class BSTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, next_state_pred, prev_state_pred, next_state_true, prev_state_true):
        next_state_loss = F.mse_loss(next_state_pred, next_state_true)
        prev_state_loss = F.mse_loss(prev_state_pred, prev_state_true)
        loss = next_state_loss + prev_state_loss
        return loss

class BeliefStateTransformer(nn.Module):
    """
    Trajectory BST
    """

    def __init__(self, state_dim, embed_dim, hidden_dim, num_layers, num_heads, context_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.fss_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.bss_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.F_encoder = GPT_Backbone(
            embed_dim=embed_dim,
            context_length=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            attn_dropout=0.1,
            block_output_dropout=0.1,
            activation="gelu"
        )
        self.B_encoder = GPT_Backbone(
            embed_dim=embed_dim,
            context_length=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            attn_dropout=0.1,
            block_output_dropout=0.1,
            activation="gelu"
        )

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.next_state = nn.Linear(hidden_dim, state_dim)
        self.prev_state = nn.Linear(hidden_dim, state_dim)

    def forward(self, f_state_seq, b_state_seq):
        """
        Note that suffix is temporally reversed

        """
        
        f_proj = self.fss_proj(f_state_seq)
        b_proj = self.bss_proj(b_state_seq)

        f_last = f_state_seq[:, -1, :] # last forward state in the sequence
        b_last = b_state_seq[:, -1, :] # last backward state in the sequence

        B, Tf, D = f_proj.shape
        _, Tb, _ = b_proj.shape

        f_proj = f_proj + self.pos_embed[:, :Tf, :]
        b_proj = b_proj + self.pos_embed[:, :Tb, :]

        f_enc = self.F_encoder(f_proj)
        b_enc = self.B_encoder(b_proj)

        f_token = f_enc[:, -1, :]
        b_token = b_enc[:, -1, :]

        belief_state = self.output_head(torch.cat((f_token, b_token), dim=-1))
        
        # outperforms directly generating next_state_pred/prev_state_pred
        next_state_pred = f_last + self.next_state(belief_state)
        prev_state_pred = b_last + self.prev_state(belief_state)

        return next_state_pred, prev_state_pred