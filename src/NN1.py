import torch.nn.functional as func
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch
import math

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

"""
But : Prédire le nombre N1 de point dans un polygone donnée

L'approcimation s'appelle N1a

Feedforward NN with multilayer perceptrons

for 
Nc edges 
input : 
Pc
ls

Loss function
L(N1,N1a)=|N1-N1A|

Algo : 
while required number of iterations is not reached do
    foreach training example in D do
        Compute N1a I using current parameters
        Calculate loss function 
        backprop 
        Update WK using Adam learning rate optimization
    end
end

du coup je dois : 
    creer un tenseur avec Ntrain elts D=(Pc,ls,N1)
    
"""


class NN1PolygonDataset(Dataset):
    """
    Dataset for NN1
    """

    def __init__(self, annotation_file: Path, polygons_dir: Path):
        self.polygons_dir = Path(polygons_dir)
        try:
            self.polygons_labels = pd.read_csv(annotation_file)
        except FileNotFoundError as err:
            print(err)
            exit(-1)

    def __len__(self):
        return len(self.polygons_labels)

    def __getitem__(self, idx):
        polygon_path = self.polygons_dir / \
            Path(self.polygons_labels.iloc[idx, 0])
        try:
            polygon = np.loadtxt(polygon_path)
        except IOError as err:
            print(err)
            exit(-1)

        N1 = self.polygons_labels.iloc[idx, 1]
        return polygon, N1


# poly{Nc}_{idx}.dat
# label
# idx,Nc


class NN1(nn.Module):

    def __init__(self, n_features: int):
        super(NN1, self).__init__()
        self.l1 = nn.Linear(n_features, 4 * n_features)
        self.b1 = nn.BatchNorm1d(4 * n_features)
        self.l2 = nn.Linear(4 * n_features, 4 * n_features)
        self.b2 = nn.BatchNorm1d(4 * n_features)
        self.l3 = nn.Linear(4 * n_features, 1)
        self.b3 = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor):
        x = self.l1(x.float())
        x = self.b1(x)
        x = func.relu(x)
        x = self.l2(x)
        x = self.b2(x)
        x = func.relu(x)
        x = self.l3(x)
        x = self.b3(x)
        return x


def train_loop(dataloader: DataLoader, model: NN1, loss_fn: nn.L1Loss, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def test_loop(dataloader: DataLoader, model: NN1, loss_fn: nn.L1Loss, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return


def main():

    # Define model's hyperparameters
    Nc = 6
    lr = 1e-4
    w = 1e-1
    batch_size = 128
    num_epochs = 1000
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # Data
    data_Path = Path(f"data/{Nc}/polygons")
    label_path = Path(f"data/{Nc}/labels")
    training_data = NN1PolygonDataset(label_path, data_Path)
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True)
    # TODO: build test data
    test_data = NN1PolygonDataset(label_path, data_Path)
    test_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True)

    # Model
    model = NN1(2 * Nc + 1)

    # Loss function
    loss = nn.L1Loss()
    # Optimizer
    opt = optim.Adam(params=model.parameters(), lr=lr, weight_decay=w)

    for epoch in tqdm(range(num_epochs)):
        train_loop(train_dataloader, model, loss, opt, device)
        test_loop(train_dataloader, model, loss, device)

    model_name = Path(f"model_{Nc}.pth")
    model_w_name = Path(f"model_weights_{Nc}.pth")
    torch.save(model, model_name)
    torch.save(model.state_dict(), model_w_name)
    return


if __name__ == "__main__":

    main()
