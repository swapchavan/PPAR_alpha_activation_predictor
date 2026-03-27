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

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class get_data:
    def __init__(self):
        # Train 
        mc_set = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/herg_train_scaled_padel_selected_descr.csv', header=0, index_col=0)
        #mc_set = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/trial_train.csv', header=0, index_col=0)
        mc_set = mc_set.replace([np.inf, -np.inf], np.nan).replace(["Infinity","-Infinity"],  np.nan).dropna(axis=1, how="any")
        mc_set_class = pd.read_csv('C:/Swapnil/herg_project/train_set_class.csv', header=0, index_col=None)
        #mc_set_class = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/trial_train_class.csv', header=0, index_col=None)
        self.X = mc_set
        self.Y = mc_set_class
        # Test
        mv_set = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/herg_test_scaled_padel_selected_descr.csv', header=0, index_col=0)
        #mv_set = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/trial_test.csv', header=0, index_col=0)
        mv_set = mv_set[mc_set.columns.to_list()]
        mv_set_class = pd.read_csv('C:/Swapnil/herg_project/test_set_class.csv', header=0, index_col=None)
        #mv_set_class = pd.read_csv('C:/Swapnil/herg_project/Deep_learning_model/Gpyopt_DNN/input_data/trial_test_class.csv', header=0, index_col=None)
        self.X_Test = mv_set
        self.Y_Test = mv_set_class
    def extract(self):
        return self.X, self.Y, self.X_Test, self.Y_Test;