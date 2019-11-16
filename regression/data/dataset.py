from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class BikeRentalDataset(Dataset):
    def __init__(self, file_path: str):

        self.data = self._read_csv(file_path)

    def _read_csv(self, file_path: str):
        data = list()

        with open(file_path) as stream:
            next(stream)
            for row in stream.readlines():
                data.append(row.split(',')[2:])
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.data)
