import sys
import csv
csv.field_size_limit(sys.maxsize)
import os
import time
import datetime
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import logging

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc 
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict
from datasets import load_from_disk

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)

def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_max_lengths(data_path):
    word_length_list = []
    dataframe=pd.read_csv(data_path)
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        next(reader,None) # skip the header
        for line in tqdm(reader,total=dataframe.shape[0], leave=True, position=0):
            
            text = ""
            for tx in line[1].split():
                text += tx.strip().lower()
                text += " "
                
            word_list = word_tokenize(text)
            word_length_list.append(len(word_list))

    sorted_word_length = sorted(word_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))]
    
def under_sampling(df_train,target_variable, seed, negative_positive_ratio):
    np.random.seed(seed)
    LABEL=df_train[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    positive_idx=IDX[LABEL==1]
    negative_idx=np.random.choice(IDX[LABEL==0],size=(len(positive_idx)*negative_positive_ratio,))
    _idx=np.concatenate([positive_idx,negative_idx])
    under_sampling_train=df_train.loc[_idx,:]
    return under_sampling_train


def mask_creation(dataset,target_variable, seed, validation_split):
    train_idx=[]
    val_idx=[]
    
    LABEL=dataset[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    target_list=np.unique(LABEL).tolist()
        
    for i in range(len(target_list)):
        _idx=IDX[LABEL==target_list[i]]
        np.random.seed(seed)
        np.random.shuffle(_idx)
        
        split=int(np.floor(validation_split*_idx.shape[0]))
        
        val_idx.extend(_idx[ : split])
        train_idx.extend(_idx[split:])        
 
    all_idx=np.arange(LABEL.shape[0])

    val_idx=np.array(val_idx)
    train_idx=np.array(train_idx)
    
    return train_idx, val_idx


def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y.squeeze()==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

def get_best_threshold(y_true, y_pred):
    best_threshold=None
    best_f1_score=0
    
    for threshold in np.arange(0.01,1,0.01):
        y_pred_class=(y_pred>=threshold).astype(int)
        f1=f1_score(y_true,y_pred_class,pos_label=1)
        
        if f1>best_f1_score:
            best_f1_score=f1
            best_threshold=threshold
    return best_threshold


def evaluate_thresholds(y_true, y_pred, thresholds, pos_label=False):
    """
    Evaluates the precision, recall, and F1 score at a fixed set of threshold values.
    Computes metrics only for positive samples with label=1.
    
    Parameters:
    y_true (ndarray): Ground truth binary labels (0 or 1).
    y_pred (ndarray): Predicted probabilities of the positive class (between 0 and 1).
    thresholds (list or ndarray): List or array of threshold values to evaluate.
    
    Returns:
    list of tuples: List of tuples containing the precision, recall, and F1 score at each threshold.
    """
    if pos_label:
        # select positive samples with label=1
        y_true_pos=y_true[y_true==1]
        y_pred_pos=y_pred[y_true==1]
    else:
        y_true_pos=y_true
        y_pred_pos=y_pred  
    
    results = []
    for threshold in tqdm(thresholds,total=thresholds.shape[0],leave=True, position=0):
        # use the threshold to make binary predictions
        y_pred_binary = (y_pred_pos >= threshold).astype(int)
        
        # compute precision, recall, and F1 score for positive samples with label=1
        if np.sum(y_pred_binary)==0:
            precision=0
        else:
            precision = precision_score(y_true_pos, y_pred_binary)
        recall = recall_score(y_true_pos, y_pred_binary)
        f1 = f1_score(y_true_pos, y_pred_binary)
        
        results.append((precision, recall, f1,threshold))
        
        result_df=pd.DataFrame(results, columns=["precision","recall","f1","threshold"])
        
    return result_df


def find_optimal_threshold(y_true, y_pred, min_recall=0.85, pos_label=False):
    
    if pos_label:
        # select positive samples with label=1
        y_true_pos=y_true[y_true==1]
        y_pred_pos=y_pred[y_true==1]
    else:
        y_true_pos=y_true
        y_pred_pos=y_pred      
    
    # precisions, recalls, thresholds=precision_recall_curve(y_true_pos, y_pred_pos)
    # f1_scores=2*(precisions*recalls)/(precisions+recalls)
    # idx=np.argmax(precisions[recalls>=0.9])
    # threshold=thresholds[recalls>=0.9][idx]
    # # Find indices of thresholds where recalls>=0.9
    # idx=np.where(recalls>=0.90)[0]
    # # Find index of threshold that maximize precision
    # opt_idx=np.argmax(precisions[idx])
    # opt_threshold=thresholds[idx][opt_idx]
    
    thresholds = np.arange(0, 1.01, 0.0001)
    result_df=evaluate_thresholds(y_true_pos, y_pred_pos, thresholds, pos_label)
    result_df.sort_values(by="recall", ascending=False, inplace=True)
    result_df=result_df[result_df["recall"]>min_recall]
    result_df.sort_values(by="precision", ascending=True, inplace=True)
    result_df=result_df.nlargest(1,"precision",keep="last")
    
    return result_df.threshold.values[0]

def eval_func(data_loader,model,device,num_classes=2,loss_weight=None):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    
    model=model.to(device)
#     for batch_idx, batch in enumerate(data_loader):
    batch_idx=0
    for batch in tqdm(data_loader, position=0, leave=True):

        te_feature = batch["input_ids"].to(device)
        te_label = batch["labels"].to(device)
        with torch.no_grad():
            te_predictions = model(te_feature,device)
            
        if loss_weight is None:
            loss = F.cross_entropy(te_predictions.view(-1, num_classes),te_label)
        else:
            loss = F.cross_entropy(te_predictions.view(-1, num_classes),te_label, weight=loss_weight.float()) 
            
        losses.append(loss.item())
        
        fin_targets.append(te_label.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(te_predictions.view(-1, num_classes),dim=1).cpu().detach().numpy())   

        batch_idx+=1

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def model_evaluate(target, y_pred):
    
    # best_threshold=find_optimal_threshold(target, predicted)
    # y_pred=[1 if x>best_threshold else 0 for x in predicted[:,1]]
    
    true_label_mask=[1 if (x-target[i])==0 else 0 for i,x in enumerate(y_pred)]
    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy=true_prediction/nb_prediction
    
    
    false_positive=np.sum([1 if v==1 and target[i]==0  else 0 for i,v in enumerate(y_pred)])
    false_negative=np.sum([1 if v==0 and target[i]==1  else 0 for i,v in enumerate(y_pred)])
    
    # precision, recall, fscore, support = precision_recall_fscore_support(target, predicted.argmax(axis=1))
    
    # precision, recall, thresholds = precision_recall_curve(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())
    precision, recall, thresholds = precision_recall_curve(target,y_pred,pos_label=1)
    pr_auc = auc(recall, precision)
    
    prec=precision_score(target,y_pred,pos_label=1)
    rec=recall_score(target,y_pred,pos_label=1)
    fscore = f1_score(target,y_pred,pos_label=1)
    roc_auc = roc_auc_score(target,y_pred)
    
    
    return {
        "total positive":sum(target),
        "false positive":false_positive,
        "false_negative":false_negative,
        "precision":prec, 
        "recall":rec, 
        "f1_score":fscore,
        "AUC":roc_auc,
        "pr_auc":pr_auc
    }

def lift_gain_eval(logit,label,topk):
    DF=pd.DataFrame(columns=["pred_score","actual_label"])
    DF["pred_score"]=logit
    DF["actual_label"]=label
    DF.sort_values(by="pred_score", ascending=False, inplace=True)
    gain={}
    for p in topk:
        N=math.ceil(int(DF.shape[0]*p))
        DF2=DF.nlargest(N,"pred_score",keep="first")
        gain[str(int(p*100))+"%"]=round(DF2.actual_label.sum()/(DF.actual_label.sum()*p),2)
    return gain
