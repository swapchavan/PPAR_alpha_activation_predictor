import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader, WeightedRandomSampler

import pandas as pd
import numpy as np
import os
import math
import random
import time

class create_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.values
        self.Y = Y.values
        self.n_samples = X.shape[0]
        
    def __getitem__(self, index):
        #return torch.tensor(self.X[index]), torch.tensor(self.Y[index])
        #return self.X[index], self.Y[index]
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.Y[index], dtype=torch.long)
        
    def __len__(self):
        return self.n_samples;
        