import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import Conv2dHub, Highway
from .. import tokenize


class SlimCNN(nn.Module):
    def __init__(
        self,
        src_max_len,
        tgt_max_len,
        vocab_size,
        hidden_size=768,
        drop_prob=0.1,
        src_max_kernel_size=32,  # < 256
        tgt_max_kernel_size=32,
    ):
        super().__init__()

        # TODO: use different parameters on code and doc?
        self.embedding = nn.Embedding(vocab_size, hidden_size, tokenize.pad_token_id)

        self.src_conv2d = Conv2dHub(src_max_kernel_size, src_max_len, hidden_size)
        self.tgt_conv2d = Conv2dHub(tgt_max_kernel_size, tgt_max_len, hidden_size)

        out_features = self.src_conv2d.out_features + self.tgt_conv2d.out_features

        self.hidden2out = nn.Linear(out_features, 1)
        self.dropout = nn.Dropout(drop_prob)

        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight.data, -0.1, 0.1)
        elif isinstance(module, nn.Conv2d):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    # param.data.uniform_(-0.1, 0.1)
                    nn.init.kaiming_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, source_ids: Tensor, target_ids: Tensor):
        """
        Inputs:
            source_ids: [batch_size, src_max_len] (value: [0, vocab_size))
            target_ids: [batch_size, tgt_max_len] (value: [0, vocab_size))

        Outputs:
            scores: batch_size (value: 0 to 1)
        """
        x = self.embedding(source_ids).unsqueeze(1)
        # [batch_size, chan(1), src_max_len, hidden_size]
        y = self.embedding(target_ids).unsqueeze(1)
        # [batch_size, chan(1), tgt_max_len, hidden_size]

        src_feats = self.src_conv2d(x)
        tgt_feats = self.tgt_conv2d(y)

        src_tgt = torch.cat([src_feats, tgt_feats], dim=1)
        # [batch_size, filter_sum*2]

        # [batch_size], Get scores [0-1]
        logits = self.hidden2out(self.dropout(src_tgt)).squeeze(1)
        scores = torch.sigmoid(logits)
        return scores
