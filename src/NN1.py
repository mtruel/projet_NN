import torch.nn.functional as func
import torch.nn as nn
import math

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
        polygon_path = self.polygons_dir / Path(self.polygons_labels.iloc[idx, 0])
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

    def forward(self, x):
        x = func.relu(self.b1(self.l1(x)))
        x = func.relu(self.b2(self.l2(x)))
        x = self.b3(self.l3(x))


def train_loop(dataloader: DataLoader, model: NN1, loss_fn, optimizer):

    return


def main():

    # Define model's hyperparameters
    lr = 1e-4
    w = 1e-1
    batch_size = 512
    epochs = 3000

    # Data
    data_Path = Path("exports")
    label_path = Path("exports/label.dat")
    training_data = NN1PolygonDataset(label_path, data_Path)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # Model
    model = NN1(4)

    return


if __name__ == "__main__":

    main()
