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
from sklearn.metrics import auc as auc_score
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict
from datasets import load_from_disk

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
    get_scheduler
)

# from accelerate import Accelerator

# accelerator = Accelerator(fp16=True)

def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))

class Loader_Creation(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 feature_name
                ):
        super().__init__()
        self.dataset=dataset
        self.tokenizer=tokenizer
        
        self.dataset=self.dataset.map(lambda x:tokenizer(x[feature_name],truncation=True,padding="max_length"), 
                                      batched=True)
        self.dataset.set_format(type="pandas")
        self.dataset=self.dataset[:]
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,index):
        _ids = self.dataset.loc[index]["input_ids"].squeeze()
        _mask = self.dataset.loc[index]["attention_mask"].squeeze()
        _target = self.dataset.loc[index]["target"].squeeze()
        
        return dict(
            input_ids=_ids,
            attention_mask=_mask,
            labels=_target
        )
    
    def collate_fn(self,batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask=torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        
        pad_token_id=self.tokenizer.pad_token_id
        keep_mask = input_ids.ne(pad_token_id).any(dim=0)
        
        input_ids=input_ids[:, keep_mask]
        attention_mask=attention_mask[:, keep_mask]
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

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


def get_best_threshold_pre_auc(y_true,y_pred):
    precision,recall,thresholds=precision_recall_curve(y_true,y_pred,pos_label=1)
    pre_auc=auc(recall,precision)
    best_threshold=thresholds[np.argmax(precision*recall)]
    
    return best_threshold

def eval_func(data_loader,model,tokenizer,device,num_classes=2,loss_weight=None,goodToken="positive",badToken="negative"):
    
    good=tokenizer.convert_tokens_to_ids(goodToken)
    bad=tokenizer.convert_tokens_to_ids(badToken)

    model.eval()
    fin_targets=[]
    fin_outputs=[]
    losses=[]
    
    model=model.to(device)
#     for batch_idx, batch in enumerate(data_loader):
    batch_idx=0
    for batch in tqdm(data_loader, position=0, leave=True):
        mask_pos=[v.tolist().index(tokenizer.mask_token_id) for v in batch["input_ids"]]
        mask_pos=torch.tensor(mask_pos).to(device)
        batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
        
        with torch.no_grad():
            outputs=model(batch['input_ids'])
        predictions = outputs[0]
        pred=predictions.gather(1, mask_pos.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,tokenizer.vocab_size)).squeeze(1) ## dim=batch_size * vocab_size
        logits=pred[:,[good,bad]] ## dim=batch_size * 2
        prob=torch.nn.functional.softmax(logits, dim=1)
        if loss_weight is None:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), batch["labels"])
        else:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device), batch["labels"], weight=loss_weight.float().to(device))
            
        losses.append(loss.item())
        
        fin_targets.append(batch["labels"].cpu().detach().numpy())
        fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())   

        batch_idx+=1

    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses

def model_evaluate(target, predicted):
    
    best_threshold=get_best_threshold(target, predicted)
    y_pred=[1 if x>best_threshold else 0 for x in predicted[:,1]].astype(int)
    
    true_label_mask=[1 if (x-target[i])==0 else 0 for i,x in y_pred]
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
