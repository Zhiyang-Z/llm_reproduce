import torch

def gen_attn_mask(size):
    """generate attention mask with shape: (size, size)"""
    mask = torch.tril(torch.ones(size, size))
    return torch.where(mask == 0, float('-inf'), float(0.0))

def gen_padding_mask(x: torch.tensor, padding_token: int):
    """generate padding mask where padding position filled with True, otherwise, False."""
    return x == padding_token