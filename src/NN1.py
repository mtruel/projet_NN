import database_gen

from torchvision.transforms import ToTensor
import os
import torch

from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

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
        polygon_path = self.polygons_dir / Path(self.polygons_labels.iloc[idx,0])
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





if __name__ == "__main__":
    data_Path = Path("exports")
    label_path = Path("exports/label.dat")
    training_data = NN1PolygonDataset(label_path,data_Path)
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    
    
    # main()
