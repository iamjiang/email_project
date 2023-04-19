import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import csv
csv.field_size_limit(sys.maxsize)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
nltk.data.path.append(nltk_data_dir)
import numpy as np
from tqdm.auto import tqdm

from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc 
from sklearn.metrics import f1_score, precision_score, recall_score

def under_sampling(df_train,target_variable, seed, negative_positive_ratio):
    np.random.seed(seed)
    LABEL=df_train[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    positive_idx=IDX[LABEL==1]
    negative_idx=np.random.choice(IDX[LABEL==0],size=(len(positive_idx)*negative_positive_ratio,))
    _idx=np.concatenate([positive_idx,negative_idx])
    under_sampling_train=df_train.loc[_idx,:]
    return under_sampling_train

def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y.squeeze()==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

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
    for te_feature, te_label in tqdm(data_loader, position=0, leave=True):
        num_sample = len(te_label)
        te_feature = te_feature.to(device)
        te_label = te_label.to(device)
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
            
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

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path, word_ratio=0.9, sent_ratio=0.5):
    word_length_list = []
    sent_length_list = []
    dataframe=pd.read_csv(data_path)
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        next(reader,None) # skip the header
        for line in tqdm(reader,total=dataframe.shape[0], leave=True, position=0):
            text = ""
            for tx in line[1].split():
                text += tx.strip().lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(word_ratio*len(sorted_word_length))], sorted_sent_length[int(sent_ratio*len(sorted_sent_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../datasets/test_df.csv")
    print (word)
    print (sent)
    
