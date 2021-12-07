import torch.nn.functional as func
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch
import math

import sys
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        loss = loss_fn(y_pred.squeeze(), y)

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
            pred = pred.squeeze()
            test_loss += loss_fn(pred, y).item()
            # print(f"=======================\n{pred} \n {y}\n")
            # print(torch.round( pred.squeeze()))
            correct += (torch.round(pred) ==
                        y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: Accuracy: {(100*correct):>0.4f}%, Avg loss: {test_loss:>8f} \n", sep=' ', end='', flush=True)
    return test_loss, correct


def main(Nc: int):

    # Define model's hyperparameters
    # Nc = 6
    lr = 1e-4
    w = 1e-1
    training_data_part = 0.8
    batch_size = 512
    num_epochs = 3000
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # Data
    data_path = Path(f"data/{Nc}")

    polygons_path = data_path / Path(f"polygons")
    label_path = data_path / Path(f"labels")

    dataset = NN1PolygonDataset(label_path, polygons_path)

    # Split dataset
    len_train = int(len(dataset)*training_data_part)
    len_test = int(len(dataset) - len_train)
    datasets_list = torch.utils.data.dataset.random_split(
        dataset, (len_train, len_test))

    training_dataset = datasets_list[0]
    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets_list[1]
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    # Model Path
    trace_path = data_path / Path("residuals")
    model_path = data_path / Path(f"model_{Nc}.pth")
    model_w_path = data_path / Path(f"model_weights_{Nc}.pth")

    losses = np.zeros(0)
    corrects = np.zeros(0)

    # Model
    # Load a pretrainned model
    try:
        print("Loading old model weights")
        trace = np.loadtxt(trace_path)
        losses = trace[:, 0]
        corrects = trace[:, 1]
        model = torch.load(model_path)
        model.load_state_dict(torch.load(model_w_path))
    except (FileNotFoundError, OSError):
        model = NN1(2 * Nc + 1)
        losses = np.zeros(0)
        corrects = np.zeros(0)
        print("Can't load old weights")

    # Loss function
    loss = nn.L1Loss()
    # Optimizer
    opt = optim.Adam(params=model.parameters(), lr=lr, weight_decay=w)

    test_loss = correct = 0.
    # Display a neat progress bar
    pbar = tqdm(range(num_epochs))
    # Plot created outside loop for efficient memory
    fig, axes = plt.subplots(nrows=3)
    for epoch in pbar:
        # Learn
        train_loop(train_dataloader, model, loss, opt, device)
        # Test
        test_loss, correct = test_loop(train_dataloader, model, loss, device)
        # Progress bar
        pbar.set_description(
            f"Epoch {epoch} / {num_epochs - 1} | Accuracy: {(100*correct):>0.4f}%, Avg loss: {test_loss:>8f}")

        # Save in trace
        losses = np.append(losses, test_loss)
        corrects = np.append(corrects, correct)

        # Plot loss and accuracy
        if epoch % 10 == 0:
            axes[0].clear()
            axes[0].set(ylabel="Loss")
            axes[0].plot(losses[-50:])

            axes[1].clear()
            axes[1].set(ylabel="Loss", yscale="log")
            axes[1].plot(losses)

            axes[2].clear()
            axes[2].set(ylabel="Accuracy")
            axes[2].plot(corrects)
            fig.savefig(data_path / Path("loss.png"))
    plt.close(fig)

    # Save accuracy and loss in residal file
    trace = np.zeros((losses.size, 2))
    trace[:, 0] = losses
    trace[:, 1] = corrects
    np.savetxt(trace_path, trace)

    # Save model weights
    torch.save(model, model_path)
    torch.save(model.state_dict(), model_w_path)
    return


def load_model(Nc: int, data_path):
    data_Path = Path(f"data/{Nc}/polygons")
    label_path = Path(f"data/{Nc}/labels")


if __name__ == "__main__":
    arguments = sys.argv
    del arguments[0]
    print(f"Arg list : {arguments}")
    for Nc in arguments:
        print(f"Learning for {Nc} boundary vertices")
        main(int(Nc))
