import torch
import numpy as np
from torch_geometric.data import Data, Dataset

class SpatioTemporalDataset(Dataset):
    """
    Spatio-temporal dataset (following EAC's SpatioTemporalDataset)
    """
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        super().__init__()
        if mode == 'default':
            self.x = inputs[split+'_x']  # (T, D, N)
            self.y = inputs[split+'_y']  # (T, 12, N)
        else:
            self.x = x
            self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)  # (N, D)
        y = torch.Tensor(self.y[index].T)  # (N, 12)
        return Data(x=x, y=y)
