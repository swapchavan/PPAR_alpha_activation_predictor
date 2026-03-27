import pandas as pd
import numpy as np
import os
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def train_final_model(my_model, EPOCHS, patience, train_iterator, val_iterator, test_iterator, ext_test_iterator, loss_fun, opt, scheduler, device, path_dir, seed_here):
    # For reproducibility
    # Python / NumPy
    random.seed(seed_here)
    np.random.seed(seed_here)
    # PyTorch
    torch.manual_seed(seed_here)
    torch.cuda.manual_seed(seed_here)
    torch.cuda.manual_seed_all(seed_here)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # cuDNN determinism
    cudnn.deterministic = True
    cudnn.benchmark = False
        
    os.chdir(path_dir)
    col_names =  ['Epoch', 'Model', 'Total_epochs','Train_LOSS',
                  'Train_BA', 'Train_Accu', 'Train_ROC_AUC_score','Train_TN', 'Train_FP', 'Train_FN', 'Train_TP',
                  'Train_SN', 'Train_SP', 'Train_precision', 'Train_MCC', 'Train_f1_score', 'Train_cohen_kappa',
                  'LOSS_Val','BA_Val', 'Accu_Val', 'ROC_score_Val', 'TN_Val','FP_Val', 'FN_Val', 'TP_Val', 'SN_Val',
                  'SP_Val', 'precision_Val','MCC_Val', 'f1_score_Val', 'cohen_kappa_Val',
                  'Test_TN', 'Test_FP', 'Test_FN', 'Test_TP', 'Test_SN', 'Test_SP', 'Test_BA', 'Test_Acu', 'Test_ROC_score', 
                  'Test_precision', 'Test_MCC', 'Test_f1_score', 'Test_cohen_kappa']
    my_output = []
    my_output = pd.DataFrame(index=range(0, EPOCHS), columns = col_names)
    #min_val_ROC = float(0.0)
    #min_val_BA = float(0.0)
    min_val_loss = float("inf")
    #max_val_BA = -1.0
    best_model_val_BA = float(0.0)
    best_model_val_ROC = float(0.0)
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        s1 = time.time()
        print(f"\nEpoch {epoch} (LR : {opt.param_groups[0]['lr']}) ____________")
        pred_class_for_epoch = np.empty((0), int)
        true_class_for_epoch = np.empty((0), int)
        pred_prob_for_epoch = np.empty((0, 2), dtype=np.float32)
        
        for i, (batch_X, batch_y) in enumerate(train_iterator):
            batch_X, batch_y = batch_X.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.long)
            #print(f'batch_X shape : {batch_X.shape} | batch_y shape : {batch_y.shape}')           
            # train model
            my_model.train()
            opt.zero_grad()
            logits = my_model(batch_X)                 # (N,2)
            batch_y = batch_y.squeeze(1).long()        # (N,)
            ce = loss_fun(logits, batch_y)
            logit_penalty = 1e-4 * (logits ** 2).mean()
            loss = ce + logit_penalty
            loss.backward()
            opt.step()
            
            # accumulate loss 
            epoch_loss += loss.item()
            n_batches += 1

            # predictions
            #predicted = my_model.predict_class(output)
            predicted = my_model.predict_class(logits)
            pred_1 = predicted.cpu().data.numpy()
            pred_class_for_epoch = np.append(pred_class_for_epoch, pred_1, axis=0)
            # true classes 
            true_class_for_epoch = np.append(true_class_for_epoch, batch_y.cpu(), axis=0)
            # pred porb
            #predicted_prob = my_model.predict_prob(output)
            predicted_prob = my_model.predict_prob(logits)
            pred_1_prob = predicted_prob.cpu().data.numpy()
            #print(f'pred_1_prob shape : {pred_1_prob.shape} \n pred_1_prob : {pred_1_prob}')
            pred_prob_for_epoch = np.append(pred_prob_for_epoch, pred_1_prob, axis=0)
        # average loss 
        epoch_loss /= max(n_batches, 1)
        
        TN, FP, FN, TP = confusion_matrix(true_class_for_epoch, pred_class_for_epoch).ravel()
        print(f' TRAIN      :: TN = {TN}, FP = {FP}, FN = {FN}, TP = {TP}')
        pred_prob_for_epoch_pos_class = pred_prob_for_epoch[:, 1]
        ROC_train = roc_auc_score(true_class_for_epoch, pred_prob_for_epoch_pos_class)
        
        #Predictions
        my_output.loc[epoch,['Epoch']] = epoch
        my_output.loc[epoch,['Model']] = 'DNN_FP_EDR_alpha_model'
        my_output.loc[epoch,['Total_epochs']] = EPOCHS
        my_output.loc[epoch,['Train_LOSS']] = epoch_loss
        my_output.loc[epoch,['Train_TN']] = TN
        my_output.loc[epoch,['Train_FP']] = FP
        my_output.loc[epoch,['Train_FN']] = FN
        my_output.loc[epoch,['Train_TP']] = TP
        train_deno = int(TP)+int(FN)
        sensitivity = int(TP)/train_deno if train_deno != 0 else 0
        #print(f'train :: sensitivity : {sensitivity}')
        my_output.loc[epoch,['Train_SN']] = sensitivity
        specificity = TN/(TN+FP)
        my_output.loc[epoch,['Train_SP']] = specificity
        BA_train = (sensitivity + TN/(TN+FP))/2
        #print(f'train :: BA_train : {BA_train}')
        my_output.loc[epoch,['Train_BA']] = BA_train
        acc_train = format((TP+TN)/(TP+FP+FN+TN))
        my_output.loc[epoch,['Train_Accu']] = acc_train
        precision = TP/(TP+FP)
        my_output.loc[epoch,['Train_ROC_AUC_score']] = format(ROC_train)
        my_output.loc[epoch,['Train_precision']] = format(precision)
        my_output.loc[epoch,['Train_MCC']] = format(matthews_corrcoef(true_class_for_epoch, pred_class_for_epoch))         # format((TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        my_output.loc[epoch,['Train_f1_score']] = format(f1_score(true_class_for_epoch, pred_class_for_epoch))    # format(2*precision*sensitivity/(precision + sensitivity))
        my_output.loc[epoch,['Train_cohen_kappa']] = format(cohen_kappa_score(true_class_for_epoch, pred_class_for_epoch))
        del(TN, FP, FN, TP, sensitivity, specificity, precision)
        
        # validate model
        val_loss=0.0
        pred_class_for_epoch_val = np.empty((0), int)
        true_class_for_epoch_val = np.empty((0), int)
        pred_prob_for_epoch_val = np.empty((0, 2), dtype=np.float32)
        
        my_model.eval()
        val_loss = 0.0
        n_val_batches = 0
        # turn off gradients for validation
        with torch.no_grad():
            for i, (batch_X_val, batch_y_val) in enumerate(val_iterator):
                #if i<3:
                batch_X_val, batch_y_val = batch_X_val.to(device, dtype=torch.float32), batch_y_val.to(device, dtype=torch.long)
                val_outputs = my_model.forward(batch_X_val)
                #batch_y_val = batch_y_val.squeeze(1) 
                batch_y_val = batch_y_val.squeeze(1).long() 
                val_loss += loss_fun(val_outputs, batch_y_val).item()
                n_val_batches += 1

                # val predictions
                predicted_val = my_model.predict_class(val_outputs)
                pred_1_val = predicted_val.cpu().data.numpy()
                pred_class_for_epoch_val = np.append(pred_class_for_epoch_val, pred_1_val, axis=0)
                # val real class
                true_class_for_epoch_val = np.append(true_class_for_epoch_val, batch_y_val.cpu(), axis=0)
                # pred porb
                predicted_prob_val = my_model.predict_prob(val_outputs)
                pred_1_prob_val = predicted_prob_val.cpu().data.numpy()
                pred_prob_for_epoch_val = np.append(pred_prob_for_epoch_val, pred_1_prob_val, axis=0)
                
        #val_loss_epoch = val_loss / len(val_iterator)
        val_loss_epoch = val_loss / max(n_val_batches, 1)
        
        #print(f'pred_class_for_epoch_val shape : {pred_class_for_epoch_val.shape}  | true_class_for_epoch_val shape : {true_class_for_epoch_val.shape}')
        TN, FP, FN, TP = confusion_matrix(true_class_for_epoch_val, pred_class_for_epoch_val).ravel()
        print(f' Validation :: TN = {TN}, FP = {FP}, FN = {FN}, TP = {TP}')
        
        pred_prob_for_epoch_val_pos_class = pred_prob_for_epoch_val[:, 1]
        ROC_val = roc_auc_score(true_class_for_epoch_val, pred_prob_for_epoch_val_pos_class)            
    
        my_output.loc[epoch,['LOSS_Val']] = val_loss_epoch
        my_output.loc[epoch,['TN_Val']] = TN
        my_output.loc[epoch,['FP_Val']] = FP
        my_output.loc[epoch,['FN_Val']] = FN
        my_output.loc[epoch,['TP_Val']] = TP
        sensitivity = TP/(TP+FN)
        #print(f'val :: sensitivity : {sensitivity}')
        my_output.loc[epoch,['SN_Val']] = sensitivity
        specificity = TN/(TN+FP)
        my_output.loc[epoch,['SP_Val']] = specificity
        BA_val = float(format((sensitivity + TN/(TN+FP))/2))
        my_output.loc[epoch,['BA_Val']] = BA_val
        acc_val = format((TP+TN)/(TP+FP+FN+TN))
        my_output.loc[epoch,['Accu_Val']] = acc_val
        precision = TP/(TP+FP)
        my_output.loc[epoch,['ROC_score_Val']] = format(ROC_val)
        my_output.loc[epoch,['precision_Val']] = format(precision)
        my_output.loc[epoch,['MCC_Val']] = format(matthews_corrcoef(true_class_for_epoch_val, pred_class_for_epoch_val)) # format((TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        my_output.loc[epoch,['f1_score_Val']] = format(f1_score(true_class_for_epoch_val, pred_class_for_epoch_val))  # format(2*precision*sensitivity/(precision + sensitivity))
        my_output.loc[epoch,['cohen_kappa_Val']] = format(cohen_kappa_score(true_class_for_epoch_val, pred_class_for_epoch_val))
        del(TN, FP, FN, TP, sensitivity, specificity, precision)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss_epoch)
            #scheduler.step(BA_val)
        else:
            scheduler.step()
        
        # Test predictions
        pred_class_for_epoch_test = np.empty((0), int)
        true_class_for_epoch_test = np.empty((0), int)
        pred_prob_for_epoch_test = np.empty((0, 2), dtype=np.float32)        

        my_model.eval()
        # turn off gradients for validation
        with torch.no_grad():
            for i, (batch_X_test, batch_y_test) in enumerate(test_iterator):
                #if i<3:
                batch_X_test, batch_y_test = batch_X_test.to(device, dtype=torch.float32), batch_y_test.to(device, dtype=torch.long)
                net_out = my_model.forward(batch_X_test)
                # test predictions
                predicted_test = my_model.predict_class(net_out)
                pred_1_test = predicted_test.cpu().data.numpy()
                pred_class_for_epoch_test = np.append(pred_class_for_epoch_test, pred_1_test, axis=0)                
                # val real class
                true_class_for_epoch_test = np.append(true_class_for_epoch_test, batch_y_test.cpu().squeeze(1), axis=0)
                # pred porb
                predicted_prob_test = my_model.predict_prob(net_out)
                pred_1_prob_test = predicted_prob_test.cpu().data.numpy()
                pred_prob_for_epoch_test = np.append(pred_prob_for_epoch_test, pred_1_prob_test, axis=0)
                
        TN, FP, FN, TP = confusion_matrix(true_class_for_epoch_test, pred_class_for_epoch_test).ravel()
        print(f' TEST        :: TN = {TN}, FP = {FP}, FN = {FN},TP  = {TP}')
        pred_prob_for_epoch_test_pos_class = pred_prob_for_epoch_test[:, 1]
        ROC_test = roc_auc_score(true_class_for_epoch_test, pred_prob_for_epoch_test_pos_class) 
        
        my_output.loc[epoch,['Test_TN']] = TN
        my_output.loc[epoch,['Test_FP']] = FP
        my_output.loc[epoch,['Test_FN']] = FN
        my_output.loc[epoch,['Test_TP']] = TP
        sensitivity = TP/(TP+FN)
        my_output.loc[epoch,['Test_SN']] = sensitivity
        specificity = TN/(TN+FP)
        my_output.loc[epoch,['Test_SP']] = specificity
        BA_test = format((sensitivity + TN/(TN+FP))/2)
        my_output.loc[epoch,['Test_BA']] = BA_test
        acc_test = format((TP+TN)/(TP+FP+FN+TN))
        my_output.loc[epoch,['Test_Acu']] = acc_test
        precision = TP/(TP+FP)
        my_output.loc[epoch,['Test_ROC_score']] = format(ROC_test)
        my_output.loc[epoch,['Test_precision']] = format(precision)
        my_output.loc[epoch,['Test_MCC']] = format(matthews_corrcoef(true_class_for_epoch_test, pred_class_for_epoch_test)) # format((TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        my_output.loc[epoch,['Test_f1_score']] = format(f1_score(true_class_for_epoch_test, pred_class_for_epoch_test)) # true_class_for_epoch_val, pred_class_for_epoch_val  # format(2*precision*sensitivity/(precision + sensitivity))
        my_output.loc[epoch,['Test_cohen_kappa']] = format(cohen_kappa_score(true_class_for_epoch_test, pred_class_for_epoch_test))
        del(TN, FP, FN, TP, sensitivity, specificity, precision)
        
        s2 = time.time()
        print(f"Time taken : {s2-s1:.4f} BA:: Train {float(BA_train)} Val {float(BA_val)}  Test {float(BA_test)} | ROC:: Train {float(ROC_train)}  Val {float(ROC_val)} Test {float(ROC_test)} | LOSS:: Train {float(epoch_loss)}  Val {float(val_loss_epoch)}")

        if val_loss_epoch < min_val_loss - 1e-6:
        #if ROC_val > min_val_ROC:
        #if BA_val > min_val_BA:
            # Save the model
            
            print(f'Validation loss decreased ({min_val_loss:.8f} --> {val_loss_epoch:.8f}).  Saving model ...')
            
            os.chdir(path_dir)
            #torch.save(my_model.state_dict(), 'checkpoint.pt')
            ckpt_path = Path(path_dir) / "checkpoint.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving checkpoint to:", ckpt_path)
            print("CWD:", os.getcwd())
            torch.save(my_model.state_dict(), ckpt_path)
            
            min_val_loss = val_loss_epoch
            
            # save ROC and BA of given pointer
            best_model_val_ROC = ROC_val
            best_model_val_BA = BA_val
            
            epochs_no_improve = 0
            
            # save probabilities
            best_model_TRAIN_pred_prob = pred_prob_for_epoch
            best_model_VAL_pred_prob = pred_prob_for_epoch_val
            best_model_TEST_pred_prob = pred_prob_for_epoch_test
            np.savetxt("best_model_TRAIN_pred_prob.csv", best_model_TRAIN_pred_prob, fmt='%1.4f', header="class_0_prob, class_1_prob", delimiter=",")
            np.savetxt("best_model_VAL_pred_prob.csv", best_model_VAL_pred_prob, fmt='%1.4f', header="class_0_prob, class_1_prob", delimiter=",")
            np.savetxt("best_model_TEST_pred_prob.csv", best_model_TEST_pred_prob, fmt='%1.4f', header="class_0_prob, class_1_prob", delimiter=",")
            
            # save actual classes 
            best_model_TRAIN_true_Y = true_class_for_epoch
            best_model_VAL_true_Y = true_class_for_epoch_val
            best_model_TEST_true_Y = true_class_for_epoch_test
            np.savetxt("best_model_TRAIN_true_Y.csv", best_model_TRAIN_true_Y, fmt='%1.4f', header="True_class", delimiter=",")
            np.savetxt("best_model_VAL_true_Y.csv", best_model_VAL_true_Y, fmt='%1.4f', header="True_class", delimiter=",")
            np.savetxt("best_model_TEST_true_Y.csv", best_model_TEST_true_Y, fmt='%1.4f', header="True_class", delimiter=",")
            
            best_model_TRAIN_pred_Y = pred_class_for_epoch
            best_model_VAL_pred_Y = pred_class_for_epoch_val
            best_model_TEST_pred_Y = pred_class_for_epoch_test
            np.savetxt("best_model_TRAIN_pred_Y.csv", best_model_TRAIN_pred_Y, fmt='%1.4f', header="True_class", delimiter=",")
            np.savetxt("best_model_VAL_pred_Y.csv", best_model_VAL_pred_Y, fmt='%1.4f', header="True_class", delimiter=",")
            np.savetxt("best_model_TEST_pred_Y.csv", best_model_TEST_pred_Y, fmt='%1.4f', header="True_class", delimiter=",")
            
            # External-Test predictions
            pred_class_for_epoch_ext_test = np.empty((0), int)
            true_class_for_epoch_ext_test = np.empty((0), int)        

            my_model.eval()
            # turn off gradients for validation
            with torch.no_grad():
                for i, (batch_X_test, batch_y_test) in enumerate(ext_test_iterator):
                    batch_X_test, batch_y_test = batch_X_test.to(device, dtype=torch.float32), batch_y_test.to(device, dtype=torch.long)
                    net_out = my_model.forward(batch_X_test)
                    # test predictions
                    predicted_test = my_model.predict_class(net_out)
                    pred_1_test = predicted_test.cpu().data.numpy()
                    pred_class_for_epoch_ext_test = np.append(pred_class_for_epoch_ext_test, pred_1_test, axis=0)                
                    # val real class
                    true_class_for_epoch_ext_test = np.append(true_class_for_epoch_ext_test, batch_y_test.cpu().squeeze(1), axis=0)
            
            # print external test set
            print('external test set predictions :')        
            for i, (true, pred) in enumerate(zip(true_class_for_epoch_ext_test, pred_class_for_epoch_ext_test)):
                print(f"Compound {i}: True={true} Predicted={pred}")
            print('\n')
            
        else:
            epochs_no_improve += 1
            print(f'EarlyStopping counter: {epochs_no_improve} out of {patience}')
        if epochs_no_improve == patience:
            early_stop = True
            print(f"Early stopping at Epoch {epoch}")
            break;
        else:
            continue;

    # load the last checkpoint with the best model
    os.chdir(path_dir)
    my_model.load_state_dict(torch.load('checkpoint.pt'))

    # Specify a path

    PATH = "entire_model.pt"

    # Save
    os.chdir(path_dir)
    torch.save(my_model, PATH)
        
    file_1 = 'FFNN_model_stats.csv'
    my_output.to_csv(file_1,header=True, sep=',')
    print('Final model construction finished!')
    
    return best_model_val_ROC;