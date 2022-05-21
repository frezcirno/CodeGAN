import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=12, batch_first=True),
            num_layers=6,
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        tgt_key_padding_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None
    ):
        r"""
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        return self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask
        )
