import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .. import tokenize


class Highway(nn.Module):
    def __init__(
        self,
        input_size,
    ):
        super(Highway, self).__init__()

        self.normal = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, input_size)

    def forward(self, x):
        y = F.relu(self.normal(x))
        t = torch.sigmoid(self.gate(x))

        return y * t + (1 - t) * x


class Conv2dHub(nn.Module):
    def __init__(self, filter_sizes, filter_nums, hidden_size, max_length):
        super(Conv2dHub, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(
                # [bat_siz x chan(1) x max_len x hid_siz]
                nn.Conv2d(1, filter_num, (filter_size, hidden_size)),
                # [bat_siz x fil_num x (max_len-fil_siz+1) x 1]
                nn.BatchNorm2d(filter_num),
                nn.ReLU(),
                nn.MaxPool2d((max_length - filter_size + 1, 1)),
                # [bat_siz x fil_num x 1 x 1]
            )
            for filter_size, filter_num in zip(filter_sizes, filter_nums)
        ])

    def forward(self, i):
        out = []
        for conv2d in self.module_list:
            # [bat_siz x fil_num x 1 x 1]
            out.append(conv2d(i).squeeze(-1).squeeze(-1))

        # [bat_siz x fil_sum]
        out = torch.cat(out, dim=1)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        src_max_len,
        tgt_max_len,
        vocab_size,
        hidden_size,
        src_drop_prob=0.1,
        tgt_drop_prob=0.1,
        max_filter_size=32,
    ):
        super(Discriminator, self).__init__()

        # TODO: use different parameters on code and doc?
        self.src_embedding = nn.Embedding(vocab_size, hidden_size, tokenize.pad_token_id)
        self.tgt_embedding = nn.Embedding(vocab_size, hidden_size, tokenize.pad_token_id)

        filter_sizes = list(range(1, max_filter_size, 4))
        filter_nums = [100 + i * 10 for i in range(1, max_filter_size, 4)]
        filter_sum = sum(filter_nums)
        # filter_sizes: [1, 5, 9, 13, 17, ...]
        # filter_nums: [110, 150, 190, ...]

        self.src_filters = Conv2dHub(filter_sizes, filter_nums, hidden_size, src_max_len)
        self.tgt_filters = Conv2dHub(filter_sizes, filter_nums, hidden_size, tgt_max_len)

        self.src_highway = Highway(filter_sum)
        self.tgt_highway = Highway(filter_sum)

        self.src_drop = nn.Dropout(src_drop_prob)
        self.tgt_drop = nn.Dropout(tgt_drop_prob)

        self.hidden2out = nn.Linear(2 * filter_sum, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias:
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
            source_ids: [batch_size x src_max_len] (value: [0, vocab_size))
            target_ids: [batch_size x tgt_max_len] (value: [0, vocab_size))

            Outputs:
            scores: batch_size (value: 0 to 1)
        """
        x = self.src_embedding(source_ids).unsqueeze(1)
        # [batch_size x chan(1) x src_max_len x hidden_size]
        y = self.tgt_embedding(target_ids).unsqueeze(1)
        # [batch_size x chan(1) x tgt_max_len x hidden_size]

        src_outputs = self.src_filters(x)
        tgt_outputs = self.tgt_filters(y)
        # [batch_size x filter_sum]

        src_outputs = self.src_highway(src_outputs)
        tgt_outputs = self.tgt_highway(tgt_outputs)

        src_outputs = self.src_drop(src_outputs)
        tgt_outputs = self.tgt_drop(tgt_outputs)
        # [batch_size x filter_sum]

        src_tgt = torch.cat([src_outputs, tgt_outputs], dim=1)
        # [batch_size x filter_sum*2]

        # [batch_size], Get scores [0-1]
        logits = self.hidden2out(src_tgt).squeeze(1)
        scores = torch.sigmoid(logits)
        return scores
