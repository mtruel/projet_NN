import torch.nn.functional as func
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torch
import math

import time
from dataclasses import dataclass
import sys
import shutil
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
"""
<<<<<<< HEAD
But : Prédire les coordonnées de N1 points dans un polygone donné

=======
But : Predire les coordonnees de N1 points dans un polygone donne
>>>>>>> 90dd57322eab0696248c69d44640c239e3365d4b
import database_gen



Feedforward NN with multilayer perceptrons

Goal: Predict the coordinates of N1 points inside a given polygonal contour.


for
Nc edges
input :
Pc
Grid
ls

Loss function
L(N1,N1a)=|N1-N1A|

Algorihtm :
while required number of iterations is not reached, do :
    for each training example in D, do :
        Compute N1a I using current parameters
        Calculate loss function
        backprop
        Update WK using Adam learning rate optimization
    end
end

du coup je dois :
    creer un tenseur avec Ntrain elts D=(Pc,ls,N1)

"""


class NN2PolygonDataset(Dataset):
    """
    Dataset for NN2
    """

    # Constructor
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
class NN2(nn.Module):

    def __init__(self, n_features: int, Np: int):
        Ngk = int(n_features/Np)
        super(NN2, self).__init__()
        self.l1 = nn.Linear(n_features, 2 * n_features + Ngk)
        self.b1 = nn.BatchNorm1d(2 * n_features + Ngk)
        self.l2 = nn.Linear(2 * n_features + Ngk, 2 * n_features + Ngk)
        self.b2 = nn.BatchNorm1d(2 * n_features + Ngk)
        self.l3 = nn.Linear(2 * n_features + Ngk, 1)
        self.b3 = nn.BatchNorm1d(Ngk)

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


def train_loop(dataloader: DataLoader, model: NN2, loss_fn: nn.L1Loss, optimizer, device):
    """Takes the training database with DataLoader and trains the NN2:
    Edits model and loss function

    :param Dataloader dataloader:
    :param NN2 model: NN2 network model
    :param nn.L1Loss loss_fn: loss function
    :param optimizer: Type of optimizer (here Adam)
    :param device: cuda or CPU

    :return:
    :rtype: np.ndarray
    """
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


def test_loop(dataloader: DataLoader, model: NN2, loss_fn: nn.L1Loss, device):
    """Takes the test database with DataLoader and tests the NN1:
    Uses model and loss function to predict

    :param Dataloader dataloader:
    :param NN2 model: NN2 network model
    :param nn.L1Loss loss_fn: loss function
    :param optimizer: Type of optimizer (here Adam)
    :param device: cuda or CPU

    :return: average of all losses
    :rtype: float
    :return: average of correct guesses
    :rtype: float
    """
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


@dataclass
class nn2_parameters:
    """
    DataClass storing parameters for the learning
    """
    # Parameters
    # Number of inner vertices
    Nc: int
    # Number of grid patches
    Np: int
    # Hyperparameters
    # Learning rate
    lr: float
    # Weight decay
    w: float
    # Size of a batch
    batch_size: int
    # Number of epoch to compute
    num_epochs: int = 100
    # Ratio of quantity of training data vs test data
    training_data_ratio: float = 0.8
    # Shuffle data
    shuffle: bool = False

    # Delete a previous model
    clean_start: bool = False
    # Paths
    # Main data folder
    data_path: Path = None
    polygons_path: Path = None
    label_path: Path = None

    trace_path: Path = None
    model_path: Path = None
    model_w_path: Path = None
    plot_path: Path = None

    # Device used for computation
    device: str = None

    # Comment about execution
    execution_notes: str = ""

    history_folder: Path = Path("history/nn2")

    log_file: Path = Path("last_executions.log")

    def __post_init__(self):
        self.lauch_date: str = time.asctime()

        # Paths
        if self.data_path is None:
            self.data_path = Path(f"data/{self.Nc}")
            self.polygons_path = self.data_path / Path(f"polygons")
            self.label_path = self.data_path / Path(f"labels_nn2")
            self.model_path = self.data_path / Path(f"model_{self.Nc}.pth")
            self.model_w_path = self.data_path / \
                Path(f"model_weights_{self.Nc}.pth")

        if self.trace_path is None:
            self.trace_path = self.data_path / Path("trace.txt")
        if self.plot_path is None:
            self.plot_path = self.data_path / Path(f"plot_{self.Nc}.png")

        if self.clean_start:
            shutil.rmtree(self.model_path, ignore_errors=True)
            shutil.rmtree(self.model_w_path, ignore_errors=True)
            shutil.rmtree(self.trace_path, ignore_errors=True)
            shutil.rmtree(self.plot_path, ignore_errors=True)

        self.current_epoch: int = 0
        # Loss and accuracy data
        self.loss_history: np.ndarray = np.zeros(0)
        self.accuracy_history: np.ndarray = np.zeros(0)
        if not(self.clean_start):
            try:
                trace = np.loadtxt(self.trace_path)
                self.loss_history = trace[:, 0]
                self.accuracy_history = trace[:, 1]
                self.current_epoch = self.loss_history.size
                print("Ancienne trace chargée")
            except (OSError):
                print("Pas d'ancienne trace")

        # Plot
        self.fig, self.axes = plt.subplots(nrows=3)

        # Define device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Header
        self.header_str = f"{self.lauch_date} | Nc={self.Nc}\n"
        self.header_str += f"        lr={self.lr}, w={self.w}, batch_size={self.batch_size}, initial_epoch={self.current_epoch}, num_epoch={self.num_epochs}, train_data_ratio={self.training_data_ratio} shuffle={self.shuffle}\n"
        self.header_str += self.execution_notes + "\n"

        # Log file
        with open(self.log_file, 'a') as f:
            f.write(self.header_str + "\n")

        # History folder
        # create tree
        try:
            self.history_folder = self.history_folder / \
                Path(f"{self.lauch_date}")
            os.makedirs(self.history_folder)
        except FileExistsError:
            pass

        return

    def add_epoch(self, avg_loss: float, accuracy: float):
        self.loss_history = np.append(self.loss_history, avg_loss)
        self.accuracy_history = np.append(self.accuracy_history, accuracy)
        self.current_epoch += 1
        return

    def update_plot(self):
        self.axes[0].clear()
        self.axes[0].set(ylabel="Avg Loss")
        self.axes[0].plot(self.loss_history[-50:])
        self.axes[0].set_title(self.header_str, {'fontsize': 6})

        self.axes[1].clear()
        self.axes[1].set(ylabel="Avg Loss", yscale="log")
        self.axes[1].plot(self.loss_history)

        self.axes[2].clear()
        self.axes[2].set(ylabel="Accuracy")
        self.axes[2].plot(self.accuracy_history)
        self.fig.savefig(self.plot_path)
        shutil.copyfile(self.plot_path, self.history_folder /
                        self.plot_path.name)
        return

    def save_trace(self):
        trace = np.zeros((self.loss_history.size, 2))
        trace[:, 0] = self.loss_history
        trace[:, 1] = self.accuracy_history
        np.savetxt(self.trace_path, trace, header=self.header_str)
        shutil.copyfile(self.trace_path, self.history_folder /
                        self.trace_path.name)
        return

    def save_model(self, model):
        torch.save(model, self.model_path)
        torch.save(model.state_dict(), self.model_w_path)
        shutil.copyfile(self.model_path, self.history_folder /
                        self.model_path.name)
        shutil.copyfile(self.model_w_path, self.history_folder /
                        self.model_w_path.name)


def train_model(parameters: nn2_parameters):
    # Chargement des données
    dataset = NN2PolygonDataset(
        parameters.label_path, parameters.polygons_path)

    len_train = int(len(dataset)*parameters.training_data_ratio)
    len_test = int(len(dataset) - len_train)
    # Split the dataset
    datasets_list = torch.utils.data.dataset.random_split(
        dataset, (len_train, len_test))

    training_dataset = datasets_list[0]
    train_dataloader = DataLoader(
        training_dataset, batch_size=parameters.batch_size, shuffle=parameters.shuffle)

    test_dataset = datasets_list[1]
    test_dataloader = DataLoader(
        test_dataset, batch_size=parameters.batch_size, shuffle=parameters.shuffle)

    # Model
    model = NN2(2 * parameters.Nc + 1, parameters.Np)
    # Load a pretrainned model
    if not(parameters.clean_start):
        try:
            model = torch.load(parameters.model_path)
            model.load_state_dict(torch.load(parameters.model_w_path))
            print("Old model weights loaded")
        except (FileNotFoundError):
            print("Can't load old weights")

    # Loss function
    loss = nn.L1Loss()
    # Optimizer
    opt = optim.Adam(params=model.parameters(),
                     lr=parameters.lr, weight_decay=parameters.w)

    # Display a neat progress bar
    pbar = tqdm(range(parameters.num_epochs))
    # Plot created outside loop for efficient memory
    fig, axes = plt.subplots(nrows=3)
    for epoch in pbar:
        # Learn
        train_loop(train_dataloader, model, loss, opt, parameters.device)
        # Test
        avg_loss, accuracy = test_loop(
            test_dataloader, model, loss, parameters.device)

        # Save in trace
        parameters.add_epoch(avg_loss, accuracy)

        # Progress bar
        pbar.set_description(
            f"Epoch {parameters.current_epoch} | Accuracy: {(100*accuracy):>0.4f}%, Avg loss: {avg_loss:>8f}")

        # save plot , trace and model
        if epoch % 10 == 1:
            parameters.update_plot()
            parameters.save_trace()
            parameters.save_model(model)

    # save plot , trace and model
    parameters.update_plot()
    parameters.save_trace()
    parameters.save_model(model)
    return


def predict():
    """Create random contour and compute predicted Nc with the loaded NN

    :return: Nc predicted number of inner nodes
    :rtype: int
    """
    Nc = 4
    ls = 1.0
    input = np.zeros(2*Nc+1)
    input[0] = ls
    coords = database_gen.create_random_contour(Nc)
    for i in range(Nc, 2):
        input[1+i] = coords[i, 0]
        input[1+i+1] = coords[i, 1]
    print(input)

    data_path = Path(f"data/{Nc}")

    model_path = data_path / Path(f"model_{Nc}.pth")
    model_w_path = data_path / Path(f"model_weights_{Nc}.pth")

    model = torch.load(model_path)
    model.load_state_dict(torch.load(model_w_path))

    model.eval()

    return model(torch.Tensor(input))  # /!\ Ne fonctionne pas /!\


if __name__ == "__main__":
    train_model(nn2_parameters(Nc=6, Np=1,
                               lr=1e-4,
                               w=1e-2,
                               batch_size=512,
                               num_epochs=5000,
                               shuffle=True,
                               clean_start=False,
                               ))
