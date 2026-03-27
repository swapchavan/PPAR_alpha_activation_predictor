from argparse import ArgumentParser
from pathlib import Path
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import shutil
import pandas as pd
import numpy as np
import os
import math
import random
import time
import joblib 

from create_dataset import create_dataset
from Net import Net
from utils import init_weights, seed_worker
#from init_weights import init_weights
#from baysian_param_calc import baysian_param_calc
from train_final_model import train_final_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  StandardScaler
from sklearn.feature_selection import VarianceThreshold

import warnings 
warnings.filterwarnings("ignore")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "1234"

def main():
    print("\n")
    parser = ArgumentParser(description="Train a DNN")
    parser.add_argument('training_set_descriptors', type=Path, help="Path to dataset")
    parser.add_argument('training_set_class', type=Path, help="Path to dataset")
    parser.add_argument('external_test_set_descriptors', type=Path, help="Path to dataset")
    parser.add_argument('external_test_set_class', type=Path, help="Path to dataset")
    parser.add_argument('n_CV', type=int, help="Number of folds for cross-validation")
    parser.add_argument('--num_workers', help="Number of parallel workers for the dataloaders", type=int, default=0)
    parser.add_argument('--device', help='which device to use', default='cpu')
    parser.add_argument('--output_dir', type=Path, help="Root directory to store results to", default=Path('models'))
    args = parser.parse_args()
    
    path_dir = args.output_dir
    n_fold_CV = args.n_CV
    
    
    # load model construction set 
    mc_set = pd.read_csv(args.training_set_descriptors, header=0, index_col=0)
    # replace infine values by Nan
    mc_set = mc_set.replace([np.inf, -np.inf], np.nan).replace(["Infinity","-Infinity"],  np.nan).dropna(axis=1, how="any")
    
    # make directory for individual fold
    descr_names_dir = os.path.join(path_dir, 'Descriptor_names')
    if not os.path.exists(descr_names_dir):
        os.makedirs(descr_names_dir)
    elif os.path.exists(descr_names_dir): 
        shutil.rmtree(descr_names_dir)
        os.makedirs(descr_names_dir)
        print('Sorry! Folder allready exists!!! OLD FOLDER HAS BEEN DELETED!!!')
    else:
        print("Sorry! CAN'T CREATE FOLDER!!!")
    os.chdir(descr_names_dir)
    
    pd.DataFrame(mc_set.columns).to_csv('Descr_names_non_inf.csv',header=False, index=False, sep=',')
    # import tox class
    mc_set_class = pd.read_csv(args.training_set_class, header=0, index_col=None)
    
    # make train & test set 
	
    mc_set, mv_set, mc_set_class, mv_set_class = train_test_split(mc_set, mc_set_class, test_size=0.1, stratify=mc_set_class, random_state=1234)
    
    # external_test set 
    ext_test_set_descr = pd.read_csv(args.external_test_set_descriptors, header=0, index_col=0)
    ext_test_set_descr = ext_test_set_descr[mc_set.columns]
    ext_test_set_class = pd.read_csv(args.external_test_set_class, header=0, index_col=None)
    
    print(f'Train shape : {mc_set.shape} | Class shape : {mc_set_class.shape} || Test data shape : {mv_set.shape} | Class shape : {mv_set_class.shape} || Ext-Test data shape : {ext_test_set_descr.shape} | Class shape : {ext_test_set_class.shape} ')
    
    # do 5 -fold CV
    device = args.device
    
    # To save final model outcome-create a empty dataframe
    col_names =  ['CV_fold', 'AUC_Val']
    final_output = []
    final_output = pd.DataFrame(index=range(1, n_fold_CV+1), columns = col_names)
    # for reproducibility 
    BASE_SEED = 1234
    # define stratified repeated k-fold CV 
    skf = StratifiedKFold(n_splits = args.n_CV, shuffle=True, random_state=BASE_SEED)
    #i = 0
    # run n_fold_CV
    #for train_indices, val_indices in skf.split(mc_set, mc_set_class):
    for i, (train_indices, val_indices) in enumerate(skf.split(mc_set, mc_set_class), start=1):	
        print('\n______________ fold : %s _______________' %i)
        #if i==2:
        
        # For reproducibility
        fold_seed = BASE_SEED + i
        random.seed(fold_seed); 
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed); 
        torch.cuda.manual_seed_all(fold_seed)
        g = torch.Generator()
        g.manual_seed(fold_seed)
        
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # cuDNN determinism
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        
        # Split 
        X_TRAIN, Y_TRAIN = mc_set.iloc[train_indices], mc_set_class.iloc[train_indices]
        X_leftover, Y_leftover = mc_set.iloc[val_indices], mc_set_class.iloc[val_indices]
        # split pre-train to train & pre_val sets
        X_Pre_VAL, X_VAL, Y_Pre_VAL, Y_VAL = train_test_split(X_leftover, Y_leftover, test_size=0.5, stratify=Y_leftover, random_state=fold_seed)
        test_set = mv_set.copy()
        test_set_class = mv_set_class.copy()
        ext_mv_set = ext_test_set_descr.copy()
        #ext_mv_set = ext_mv_set[X_TRAIN.columns].copy()
        ext_mv_set_class = ext_test_set_class.copy()
        print('\nBefore variance threshold --------------')
        print(f'Training : {X_TRAIN.shape} | {Y_TRAIN.shape}')
        print(f'Pre-validation : {X_Pre_VAL.shape} | {Y_Pre_VAL.shape}')
        print(f'Validation : {X_VAL.shape} | {Y_VAL.shape}')
        #print(f'Test : {mv_set.shape} | {mv_set_class.shape}')
        print(f'Test : {test_set.shape} | {test_set_class.shape}')
        print(f'External-Test : {ext_mv_set.shape} | {ext_mv_set_class.shape}\n')
        
        # make directory for individual fold
        fold_nm = 'CV_fold_' + str(i)
        CV_dir = os.path.join(path_dir, fold_nm)
        if not os.path.exists(CV_dir):
            os.makedirs(CV_dir)
        elif os.path.exists(CV_dir): #and os.path.is_dir(CV_dir):
            shutil.rmtree(CV_dir)
            os.makedirs(CV_dir)
            print('Sorry! Folder allready exists!!! OLD FOLDER HAS BEEN DELETED!!!')
        else:
            print("Sorry! CAN'T CREATE FOLDER!!!")
        os.chdir(CV_dir)

        # train test index saving
        pd.DataFrame(X_TRAIN.index).to_csv('X_TRAIN_ind.csv',header=False, index=False, sep=',')
        pd.DataFrame(X_Pre_VAL.index).to_csv('X_Pre_VAL_ind.csv',header=False, index=False, sep=',')
        pd.DataFrame(X_VAL.index).to_csv('X_VAL_ind.csv',header=False, index=False, sep=',')
        pd.DataFrame(test_set.index).to_csv('mv_set_ind.csv',header=False, index=False, sep=',')
        
        
        # 1) scaling
        print('scaling...')
        scaler_1 = MinMaxScaler().fit(X_TRAIN.values)
        #scaler_1 =  StandardScaler().fit(X_TRAIN.values)
        X_TRAIN = pd.DataFrame(scaler_1.transform(X_TRAIN.values), columns=X_TRAIN.columns, index=X_TRAIN.index)
        X_Pre_VAL = pd.DataFrame(scaler_1.transform(X_Pre_VAL.values), columns=X_Pre_VAL.columns, index=X_Pre_VAL.index)
        X_VAL = pd.DataFrame(scaler_1.transform(X_VAL.values), columns=X_VAL.columns, index=X_VAL.index)
        test_set = pd.DataFrame(scaler_1.transform(test_set.values), columns=test_set.columns, index=test_set.index)
        ext_mv_set = pd.DataFrame(scaler_1.transform(ext_mv_set.values), columns=ext_mv_set.columns, index=ext_mv_set.index)
        
        # create new folder & save scaler
        scaler_dir = os.path.join(CV_dir, 'scaler')
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        elif os.path.exists(scaler_dir):
            shutil.rmtree(scaler_dir)
            os.makedirs(scaler_dir)
            print('Sorry! Folder allready exists!!! OLD FOLDER HAS BEEN DELETED!!!')
        else:
            print("Sorry! CAN'T CREATE FOLDER!!!")
        os.chdir(scaler_dir)
        # save descr names & scaler 
        pd.DataFrame(X_TRAIN.columns).to_csv('Descr_for_scaling.csv',header=False, index=False, sep=',')
        joblib.dump(scaler_1,'MinMaxScaler.joblib')
        
        # 2) Variance filtering (fit on TRAIN only)
        print('after variance threshold --------------')
        vt = VarianceThreshold(threshold=0.005)
        vt.fit(X_TRAIN)
        selected_cols = X_TRAIN.columns[vt.get_support()].tolist()
        X_TRAIN = X_TRAIN.loc[:, selected_cols]
        X_Pre_VAL = X_Pre_VAL.loc[:, selected_cols]
        X_VAL = X_VAL.loc[:, selected_cols]
        test_set = test_set.loc[:, selected_cols]
        ext_mv_set = ext_mv_set.loc[:, selected_cols]
        print(f'Training : {X_TRAIN.shape} | {Y_TRAIN.shape}')
        print(f'Pre-validation : {X_Pre_VAL.shape} | {Y_Pre_VAL.shape}')
        print(f'Validation : {X_VAL.shape} | {Y_VAL.shape}')
        print(f'Test : {test_set.shape} | {test_set_class.shape}')
        print(f'External-Test : {ext_mv_set.shape} | {ext_mv_set_class.shape}\n')
        
        # create new folder & save scaler
        var_thre_descr_dir = os.path.join(CV_dir, 'var_threshold_select_descr')
        if not os.path.exists(var_thre_descr_dir):
            os.makedirs(var_thre_descr_dir)
        elif os.path.exists(var_thre_descr_dir):
            shutil.rmtree(var_thre_descr_dir)
            os.makedirs(var_thre_descr_dir)
            print('Sorry! Folder allready exists!!! OLD FOLDER HAS BEEN DELETED!!!')
        else:
            print("Sorry! CAN'T CREATE FOLDER!!!")
        os.chdir(var_thre_descr_dir)
        # save descr names  
        pd.DataFrame(selected_cols).to_csv('Selected_desc_variance_thre.csv',header=False, index=False, sep=',')
        
        # n_features
        n_features = X_TRAIN.shape[1]

        # create dataset, derive sampler and make data-iterator
        TRAIN_dataset = create_dataset(X_TRAIN, Y_TRAIN)
        PRE_VAL_dataset = create_dataset(X_Pre_VAL, Y_Pre_VAL)
        VAL_dataset = create_dataset(X_VAL, Y_VAL)
        TEST_dataset = create_dataset(test_set, test_set_class)
        EXT_TEST_dataset = create_dataset(ext_mv_set, ext_mv_set_class)
        
        # Derive weights and construct a sampler for TRAINING SET
        targets = torch.tensor(Y_TRAIN.values)
        class_sample_count = torch.tensor([(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True, generator=g)

        # ---------------- define hyperparameter --------------------------
        
        my_lr, my_bs, my_n_lay, my_neurons, my_dropout = 0.0001, 64, 1, 256, 0.2
        
        print(f'----> my_lr : {my_lr} | my_bs : {my_bs} | my_n_lay : {my_n_lay} | my_neurons : {my_neurons} | my_dropout : {my_dropout}')

        # ---------------- Final model construction ----------------------
        
        # Data loading:
        BATCH_SIZE = int(my_bs)
        train_loader = torch.utils.data.DataLoader(TRAIN_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, worker_init_fn=seed_worker, generator=g, pin_memory=True, drop_last=False)
        pre_val_loader = torch.utils.data.DataLoader(PRE_VAL_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(VAL_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(TEST_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)
        ext_test_loader = torch.utils.data.DataLoader(EXT_TEST_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)

        # Model definition:
        my_model = Net(n_features, int(my_n_lay), int(my_neurons), my_dropout)
        my_model.apply(init_weights).to(device)
        print(my_model)

        # loss fun & optimizer
        targets = torch.tensor(Y_TRAIN.values)
        class_counts = torch.bincount(targets.squeeze().long())
        class_weights = (class_counts.sum() / class_counts).float().to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(my_model.parameters(), lr=float(my_lr), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        # ----------------- Final model training -------------------------
        # make subfolder for final model
        CV_dir_subfolder_2 = os.path.join(CV_dir, 'Final_model')
        if not os.path.exists(CV_dir_subfolder_2):
            os.makedirs(CV_dir_subfolder_2)
        elif os.path.exists(CV_dir_subfolder_2):
            shutil.rmtree(CV_dir_subfolder_2)
            os.makedirs(CV_dir_subfolder_2)
        # train
        AUC_Val = train_final_model(my_model, 500, 20, train_loader, val_loader, test_loader, ext_test_loader, loss_function, optimizer, scheduler, device, CV_dir_subfolder_2, fold_seed)
        # save and store
        final_output.loc[i,['CV_fold']] = i
        final_output.loc[i,['AUC_Val']] = AUC_Val
        os.chdir(path_dir)
        final_output.to_csv('CV_results.csv',header=True, sep=',')
    
if __name__ == '__main__':
    main()