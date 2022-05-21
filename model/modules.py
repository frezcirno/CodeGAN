import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    def __init__(self, max_kernel_size, max_length, hidden_size):
        super(Conv2dHub, self).__init__()
        self.max_length = max_length
        self.max_kernel_size = max_kernel_size

        # kernel_sizes: [1, 5, 9, 13, 17, ...]
        # out_features: [110, 150, 190, ...]
        kernel_sizes = list(range(1, max_kernel_size, 4))
        out_features = [100 + i * 10 for i in kernel_sizes]

        self.out_features = sum(out_features)
        self.module_list = nn.ModuleList([
            nn.Sequential(
                # [bat_siz, chan(1), max_length, hid_siz]
                nn.Conv2d(1, out_feature, (kernel_size, hidden_size)),
                # [bat_siz, out_feature, (max_length-kernel_size+1), 1]
                nn.BatchNorm2d(out_feature),
                nn.ReLU(),
                nn.MaxPool2d((max_length - kernel_size + 1, 1)),
                # [bat_siz, out_feature, 1, 1]
            )
            for kernel_size, out_feature in zip(kernel_sizes, out_features)
        ])

    def forward(self, input: Tensor):
        """
        Input:
            input: [batch_size, chan(1), max_length, hid_siz]
        Output:
            out: [batch_size, out_features]
        """
        out = []
        for module in self.module_list:
            # [bat_siz, fil_num, 1, 1]
            out.append(module(input).squeeze(-1).squeeze(-1))

        out = torch.cat(out, dim=1)
        return out
