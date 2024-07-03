import torch
from torch.utils.data import Dataset, DataLoader

import pickle

class NuScenesCustom(Dataset):
    def __init__(self, path):
        self.path = path

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def unflatten_actions(self):

    
    def __getitem__(self, idx):
        #unflatten actions
        return self.data[idx]

nusc_data = NuScenesCustom()
nusc_data[0]