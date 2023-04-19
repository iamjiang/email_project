import os
import time
import datetime
import math
import random
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
from sklearn.metrics import classification_report, confusion_matrix

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.utils.benchmark as benchmark

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

import transformers
print("Transformers version is {}".format(transformers.__version__))

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_scheduler
)


import utils


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main(args,train_data, val_data, test_data, device):

    train_df=Dataset.from_pandas(train_data)
    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    
    model_path=args.model_name.split("-")[0]+"_"+args.model_name.split("-")[1]+"_"+"repo"

    model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/fine-tune-LM",model_path)
        
    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
            
    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()
    
    test_module=utils.Loader_Creation(test_df, tokenizer,args.feature_name)

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.test_batch_size,
                                collate_fn=test_module.collate_fn
                               )

    # %pdb
    # next(iter(train_dataloader))

    print()
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    
    train_label=train_data['target'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        

    output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/email-complaints/fine-tuning/updated-fine-tune/", args.output_dir)
    
    config=AutoConfig.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    def eval_func(data_loader,model,device,num_classes=2,loss_weight=None):
        
        fin_targets=[]
        fin_outputs=[]
        losses=[]
        model.eval()
        if args.multiple_gpus:
            model=model.to(device[0])
            model=torch.nn.DataParallel(model,device_ids=device)
        else:
            model=model.to(device)
        start_time=time.time()
        batch_idx=0
        for batch in tqdm(data_loader, position=0, leave=True):
            if not args.multiple_gpus:
                batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            else:
                batch={k:v.type(torch.LongTensor).to(device[0]) for k,v in batch.items()}
                
            inputs={k:v  for k,v in batch.items() if k!="labels"}
            with torch.no_grad():
                outputs=model(**inputs)
            logits=outputs['logits']
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes), batch["labels"])
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes), batch["labels"], weight=loss_weight.float())

            losses.append(loss.item())

            fin_targets.append(batch["labels"].cpu().detach().numpy())
            fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())   

            batch_idx+=1
            
        end_time=time.time()
        duration=end_time-start_time
        return duration 
    
    total_inputs=test_df.shape[0]
    duration=eval_func(test_dataloader,model,device,num_classes=num_classes,loss_weight=loss_weight)
    throughput=total_inputs/duration
    latency=duration/total_inputs # multiply 1000 to convert second into milliseconds per input
    
    latency_dir="/opt/omniai/work/instance1/jupyter/email-complaints/fine-tuning/updated-fine-tune/"
    if args.multiple_gpus:
        with open(os.path.join(latency_dir,"latency","throughput.txt"),'a') as f:
            f.write(f'{args.model_name},{latency},{throughput},"multiple-gpus"\n')
        print()  
        print("model_name: {:} | latency: {:.4f} | throughput: {:.4f} | device: {:} ".format(args.model_name, latency, throughput, "multiple-gpus"))    
        print()
            
    else:
        
        with open(os.path.join(latency_dir,"latency","throughput.txt"),'a') as f:
            f.write(f'{args.model_name},{latency},{throughput},{device}\n')
        print()  
        print("model_name: {:} | latency: {:.4f} | throughput: {:.4f} | device: {:} ".format(args.model_name, latency, throughput, device))    
        print()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--test_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in validation")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument('--multiple_gpus', action="store_true", help="use multiple gpus or not")
    parser.add_argument('--gpus', type=int, default=[], nargs='+', help='used gpu')
    
    args= parser.parse_args()

    args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_repo"


    seed_everything(args.seed)

    print()
    print(args)
    print()
    

    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","datasets")
    
    train_val_test=pd.read_pickle(os.path.join(data_dir,"train_val_test_pickle"))
    train_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="training_set"])
    val_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="validation_set"])
    test_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="test_set"])
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])
    
    def binary_label(example):
        return {"target": 1 if example["is_complaint"]=="Y" else 0}

    train_df=hf_data["train"].map(binary_label)
    val_df=hf_data["validation"].map(binary_label)
    test_df=hf_data["test"].map(binary_label)
        
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    
#     if torch.cuda.is_available():    
#         device = torch.device("cuda")
#         print()
#         print('{:<30}{:<10}'.format("The # of availabe GPU(s): ",torch.cuda.device_count()))

#         for i in range(torch.cuda.device_count()):
#             print('{:<30}{:<10}'.format("GPU Name: ",torch.cuda.get_device_name(i)))

#     else:
#         print('No GPU available, using the CPU instead.')
#         device = torch.device("cpu")
    
    train_data=hf_data['train'].shuffle(seed=101).select(range(len(hf_data["train"])))
    # train_data=hf_data['train'].shuffle(seed=101).select(range(500))
    train_data.set_format(type="pandas")
    train_data=train_data[:]
    
    val_data=hf_data['validation'].shuffle(seed=101).select(range(len(hf_data["validation"])))
    # val_data=hf_data['validation'].shuffle(seed=101).select(range(500))
    val_data.set_format(type="pandas")
    val_data=val_data[:]

    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    # test_data=hf_data['test'].shuffle(seed=101).select(range(300))
    test_data.set_format(type="pandas")
    test_data=test_data[:]

    if args.multiple_gpus:
        device=[torch.device(f"cuda:{i}") for i in args.gpus]
    else:
        device=torch.device(args.device)
        
    main(args,train_data, val_data, test_data, device)
    