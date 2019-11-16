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

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)


def train(model: VanillaNet):
    pass


def evaluate(model: VanillaNet):
    pass


if __name__ == '__main__':
    loader = get_loader(dataset_file_path=os.path.join(os.getcwd(), 'data', 'hour.csv'))
    pass

    # net = VanillaNet(in_channels=14)
    # train(net)
    # evaluate(net)
