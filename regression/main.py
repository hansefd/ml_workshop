import time
from fnmatch import fnmatch

import torch
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import BikeRentalDataset
from vanilla_net import VanillaNet
from sklearn.metrics import r2_score

torch.manual_seed(1)  # reproducible
BATCH_SIZE = 64
EPOCH = 10


def get_loader(dataset_file_path: str):
    dataset = BikeRentalDataset(dataset_file_path)

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)


def train(model: VanillaNet, checkpoint_folder_path: str):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = torch.nn.MSELoss()
    loader = get_loader(dataset_file_path=os.path.join(os.getcwd(), 'data', 'hour.csv'))

    model.train()
    best_prec = 0

    for _ in tqdm(range(EPOCH), total=EPOCH):
        for index, (x, y) in enumerate(loader):
            y_pred = model(x)

            loss = loss_func(y_pred.view(-1), y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            prec = r2_score(y, y_pred.detach())
            if prec > best_prec:
                best_prec = prec
                replace_checkpoint(checkpoint_folder_path, model, int(best_prec * 100))


def replace_checkpoint(checkpoint_folder_path: str, model: VanillaNet, name_suffix: int):
    models_to_replace = list(filter(lambda x: ".pth" in x, os.listdir(checkpoint_folder_path)))

    if len(models_to_replace) == 1:
        os.unlink(os.path.join(checkpoint_folder_path, models_to_replace[0]))

    model_name = f'vanilla_net_{name_suffix}.pth'
    model_path = os.path.join(checkpoint_folder_path, model_name)
    torch.save(model.state_dict(), model_path)


def evaluate(model: VanillaNet):
    model.eval()

    with torch.no_grad():
        x, y = [], []

        with open(os.path.join(os.getcwd(), 'data', 'single_valid.csv')) as stream:
            row = stream.readlines()[0]
            array = np.asarray(row.split(',')[2:], dtype=np.float32)
            tensor = torch.from_numpy(array)
            x, y = tensor[:-1], tensor[-1]

        predict = model(x)

        print(f"Predicted value is {predict.item():.2f}. Actual value is {y.item():.2f}")


def prepare_checkpoint_dir() -> str:
    root = os.path.join(os.getcwd(), 'data')
    dir_name = time.strftime("%Y%m%d-%H%M")
    dir_path = os.path.join(root, dir_name)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


if __name__ == '__main__':
    # net = VanillaNet(in_channels=14)
    # dir_path = prepare_checkpoint_dir()
    # train(net, dir_path)

    saved_model = VanillaNet(in_channels=14)
    state_dict = torch.load(os.path.join(os.getcwd(), 'data', '20191128-1534', 'vanilla_net_99.pth'))
    saved_model.load_state_dict(state_dict)

    evaluate(saved_model)
