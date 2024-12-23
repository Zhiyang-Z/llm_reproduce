import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Encoder_block(nn.Module):
    def __init__(self,
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        self.pre_norm = pre_norm
        self.self_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(ndim)
        self.ff1 = nn.Linear(ndim, ndim_feedforward)
        self.ff2 = nn.Linear(ndim_feedforward, ndim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.norm2 = nn.LayerNorm(ndim)
    def forward(self, x, attn_mask, padding_mask):
        # phase 1: self-attention
        if self.pre_norm: x = self.norm1(x)
        x = x + self.self_attention(x,x,x,attn_mask=attn_mask,key_padding_mask=padding_mask)
        if not self.pre_norm: x = self.norm1(x)
        # phase 2: feed forward
        if self.pre_norm: x = self.norm2(x)
        x = x + self.dropout2(self.ff2(self.dropout1(F.relu(self.ff1(x)))))
        if not self.pre_norm: x = self.norm2(x)
        return x

class Decoder_block(nn.Module):
    def __init__(self,
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        self.pre_norm = pre_norm
        self.self_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(ndim)
        self.cross_attention = nn.MultiheadAttention(ndim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(ndim)
        self.ff1 = nn.Linear(ndim, ndim_feedforward)
        self.ff2 = nn.Linear(ndim_feedforward, ndim)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.norm3 = nn.LayerNorm(ndim)
    def forward(self, x, encoder_out, attn_mask, padding_mask):
        # phase 1: self-attention
        if self.pre_norm: x = self.norm1(x)
        x = x + self.self_attention(x,x,x,attn_mask=attn_mask,key_padding_mask=padding_mask)
        if not self.pre_norm: x = self.norm1(x)
        # phase 2: cross-attention
        if self.pre_norm: x = self.norm2(x)
        x = x + self.cross_attention(x, encoder_out, encoder_out, attn_mask=attn_mask, key_padding_mask=padding_mask) # ????mask only take effect in the first layer???
        if not self.pre_norm: x = self.norm2(x)
        # phase 3: feed forward
        if self.pre_norm: x = self.norm3(x)
        x = x + self.dropout2(self.ff2(self.dropout1(F.relu(self.ff1(x)))))
        if not self.pre_norm: x = self.norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 mode='all',
                 nlayer=[12,12],
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        # sanity check
        if mode not in ['all', 'encoder_only', 'decoder_only']: raise ValueError("Unsupported mode.")
        if ndim % nhead != 0: raise ValueError("ndim must be divisible by nhead")

        self.vocabulary_size, self.mode, self.nhead, self.ndim, self.ndim_feedforward = vocabulary_size, mode, nhead, ndim, ndim_feedforward
        if self.mode == 'all': self.nlayer_encoder, self.nlayer_decoder = nlayer[0], nlayer[1]
        elif self.mode == 'encoder_only': self.nlayer_encoder, self.nlayer_decoder = nlayer[0], None
        else: self.nlayer_encoder, self.nlayer_decoder = None, nlayer[0]

        if self.mode == 'all':
            self.encoder_layers = nn.ModuleList([Encoder_block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               self.drop_out,
                                                               self.pre_norm) for _ in range(self.nlayer_encoder)])
            self.decoder_layers = nn.ModuleList([Decoder_block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               self.drop_out,
                                                               self.pre_norm) for _ in range(self.nlayer_decoder)])
        else:
            self.encoder_layers = nn.ModuleList([Encoder_block(self.nhead,
                                                               self.ndim,
                                                               self.ndim_feedforward,
                                                               self.drop_out,
                                                               self.pre_norm) for _ in range(self.nlayer_encoder if self.mode == 'encoder_only' else self.nlayer_decoder)])
            
        self.out = nn.Linear(self.ndim, self.vocabulary_size)
    def forward(self, encoder):
        # To-DO

