import torch
import torch.nn as nn
from Transformer import Transformer
import utils

class GPT(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 embedding_size,
                 nlayer=[12,12],
                 nhead=12,
                 ndim=768,
                 ndim_feedforward=2048,
                 drop_out=0.1,
                 pre_norm=False):
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # decoder_only transformer
        self.model = Transformer(vocabulary_size=vocabulary_size,
                                 mode='decoder_only',
                                 nlayer=nlayer,
                                 nhead=nhead,
                                 ndim=ndim,
                                 ndim_feedforward=ndim_feedforward,
                                 drop_out=drop_out,
                                 pre_norm=pre_norm)
        
    def forward(self, seq, need_padding=True):
        """seq shape: [batch, seq_length]"""
        assert seq.ndim == 2
        batch_size, seq_len = seq.shape[0], seq.shape[1]

        seq_embedding = self.embedding(seq)
        out = self.model(encoder_in=None,
                         decoder_in=seq_embedding,
                         encoder_attn_mask=None,
                         decoder_attn_maske=utils.gen_attn_mask(),
                         encoder_padding_mask=None,
                         decoder_padding_mask=utils.gen_padding_mask() if need_padding else None)
        
        return out
                                          