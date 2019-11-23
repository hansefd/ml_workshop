from fnmatch import fnmatch

import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import BikeRentalDataset
from vanilla_net import VanillaNet

torch.manual_seed(1)  # reproducible
BATCH_SIZE = 64
EPOCH = 25


def get_loader(dataset_file_path: str):
    dataset = BikeRentalDataset(dataset_file_path)

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)


def train(model: VanillaNet):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.MSELoss()
    loader = get_loader(dataset_file_path=os.path.join(os.getcwd(), 'data', 'hour.csv'))

    model.train()

    for _ in tqdm(range(EPOCH), total=EPOCH):
        for x, y in enumerate(loader):
            y_pred = model(x)

            loss = loss_func(y_pred.view(-1), y)
            loss.backward()
            optim.step()
            optim.zero_grad()


def evaluate(model: VanillaNet):
    pass


if __name__ == '__main__':
    net = VanillaNet(in_channels=14)
    train(net)
    # for x_batch, y_batch in loader:
    #     print(x_batch)
    #     print(y_batch)


    #evaluate(net)
