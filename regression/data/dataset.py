import os
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

        return np.asarray(data, dtype=np.float32)

    def __getitem__(self, index):
        tensor = torch.from_numpy(self.data[index])
        return tensor[:-1], tensor[-1]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = BikeRentalDataset(os.path.join(os.getcwd(), 'hour.csv'))

    for row in ds:
        print(row)
