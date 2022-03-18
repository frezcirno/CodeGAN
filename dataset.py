from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset
from typing import Sequence


def SamplingSubset(dataset: Dataset, num_samples: int) -> Dataset:
    indices = torch.randperm(len(dataset))[:num_samples]
    return Subset(dataset, indices)


class SelectDataset(Dataset):

    def __init__(self, dataset: Dataset, columns: Sequence[int]) -> None:
        super(SelectDataset, self).__init__()
        self.dataset = dataset
        self.columns = columns

        assert all(column in range(len(
            self.dataset[0])) for column in columns), 'columns overflow'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return tuple(col
                     for i, col in enumerate(self.dataset[idx])
                     if i in self.columns)


class CombineDataset(Dataset):

    def __init__(self, *datasets: Dataset) -> None:
        super(CombineDataset, self).__init__()
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return tuple(el for dataset in self.datasets for el in dataset[idx])


class ConstDataset(Dataset):

    def __init__(self, value: Tensor, size: int) -> None:
        super(ConstDataset, self).__init__()
        self.value = value
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size or idx < -self.size:
            raise IndexError("index out of range")
        return (self.value.clone(),)
