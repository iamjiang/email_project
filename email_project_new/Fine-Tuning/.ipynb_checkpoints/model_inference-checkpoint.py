import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v4_new_email/Fine-Tuning/')
sys.path=list(set(sys.path))

import csv
import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
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
from torch.utils.data.sampler import SubsetRandomSampler

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

import transformers
print("Transformers version is {}".format(transformers.__version__))

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from accelerate import Accelerator

import utils

from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class Loader_Creation(torch.utils.data.Dataset):
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
        _snapshot_id=self.dataset.loc[index]["snapshot_id"].squeeze()
        _thread_id=self.dataset.loc[index]["thread_id"].squeeze()
        _is_feedback=self.dataset.loc[index]["is_feedback"].squeeze()
        _text_length=self.dataset.loc[index]["text_length"].squeeze()
        
        return dict(
            input_ids=_ids,
            attention_mask=_mask,
            labels=_target,
            snapshot_id=_snapshot_id,
            thread_id=_thread_id,
            is_feedback=_is_feedback,
            text_length=_text_length
        )

    
    def collate_fn(self,batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask=torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        snapshot_id=torch.stack([torch.tensor(x["snapshot_id"]) for x in batch])
        thread_id=torch.stack([torch.tensor(x["thread_id"]) for x in batch])
        is_feedback=torch.stack([torch.tensor(x["is_feedback"]) for x in batch])
        text_length=torch.stack([torch.tensor(x["text_length"]) for x in batch])
        
        pad_token_id=self.tokenizer.pad_token_id
        keep_mask = input_ids.ne(pad_token_id).any(dim=0)
        
        input_ids=input_ids[:, keep_mask]
        attention_mask=attention_mask[:, keep_mask]
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            snapshot_id=snapshot_id,
            thread_id=thread_id,
            is_feedback=is_feedback,
            text_length=text_length
        )
    
def val_mask_creation(dataset,target_variable, validation_split):
    
    dataset.sort_values(by='time', ascending=False, axis=0, inplace = True)
    dataset=dataset.reset_index(drop=True)
    
    train_idx=[]
    val_idx=[]
    
    LABEL=dataset[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    target_list=np.unique(LABEL).tolist()
        
    for i in range(len(target_list)):
        
        _idx=IDX[LABEL==target_list[i]]
        ## split train and valiation by time instead of randomly
        # np.random.seed(seed)
        # np.random.shuffle(_idx)
        
        split=int(np.floor(validation_split*_idx.shape[0]))
        
        val_idx.extend(_idx[ : split])
        train_idx.extend(_idx[split:])        
 
    all_idx=np.arange(LABEL.shape[0])

    val_idx=np.array(val_idx)
    train_idx=np.array(train_idx)
    
    df_train=dataset.loc[train_idx,:]
    df_train=df_train.reset_index(drop=True)
    
    df_val=dataset.loc[val_idx,:]
    df_val["data_type"]=["val"]*val_idx.shape[0]
    df_val=df_val.reset_index(drop=True)
    
    return df_train, df_val

    
def main(args, val_data, test_data, test_snapshot_map, test_thread_map, device):

    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    

    model_name=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/Fine-Tuning", "results", args.test_date, args.output_dir)
            
    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=args.model_max_length)
    if args.model_name=="bigbird-roberta-large":
        config.block_size=16
        config.num_random_blocks=2
        config.attention_type="original_full"
        model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(model_name)
          
    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()

    val_module=Loader_Creation(val_df, tokenizer,args.feature_name)
    val_dataloader=DataLoader(val_module,
                              shuffle=False,
                              batch_size=args.batch_size,
                              collate_fn=val_module.collate_fn
                              )
    
    test_module=Loader_Creation(test_df, tokenizer,args.feature_name)
    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=test_module.collate_fn
                               )

    print()
    print('{:<30}{:<10,} '.format("val mini-batch",len(val_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    print()
    
    accelerator = Accelerator(fp16=args.fp16)
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    if accelerator.is_main_process:
        accelerator.print("")
        logger.info(f'Accelerator Config: {acc_state}')
        accelerator.print("")
        
    model,  val_dataloader, test_dataloader = accelerator.prepare(model, val_dataloader, test_dataloader)
    
    def eval_func(data_loader,model,device,num_classes=2):
        
        snapshot_id=[]
        thread_id=[]
        is_feedback=[]
        text_length=[]
        
        fin_targets=[]
        fin_outputs=[]
        losses=[]
        model.eval()
        if args.multiple_gpus:
            model=model.to(device[0])
            model=torch.nn.DataParallel(model,device_ids=device)
        else:
            model=model.to(device)

        batch_idx=0
        for batch in tqdm(data_loader, position=0, leave=True):
            if not args.multiple_gpus:
                batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            else:
                batch={k:v.type(torch.LongTensor).to(device[0]) for k,v in batch.items()}
                
            inputs={k:v  for k,v in batch.items() if k in ["input_ids","attention_mask"]}
            with torch.no_grad():
                outputs=model(**inputs)
            logits=outputs['logits']

            loss = F.cross_entropy(logits.view(-1, num_classes), batch["labels"])
   
            losses.append(loss.item())

            fin_targets.append(batch["labels"].cpu().detach().numpy())
            fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())  
            
            snapshot_id.append(batch["snapshot_id"].cpu().detach().numpy())
            thread_id.append(batch["thread_id"].cpu().detach().numpy())
            is_feedback.append(batch["is_feedback"].cpu().detach().numpy())
            text_length.append(batch["text_length"].cpu().detach().numpy())

            batch_idx+=1
            
        return np.concatenate(fin_outputs), np.concatenate(fin_targets), np.concatenate(snapshot_id) , np.concatenate(thread_id) , np.concatenate(is_feedback), np.concatenate(text_length)
    
    val_pred,val_target,_,_,_,_=eval_func(val_dataloader,model, device)
    test_pred,test_target,test_snapshot_id,test_thread_id,test_feedback,test_text_length=eval_func(test_dataloader,model, device)
    
    output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/Fine-Tuning", "results", args.test_date, args.output_dir)
    
    # with open(os.path.join(output_dir,"metrics_val.txt"),"r") as file:
    #     for line in file:
    #         best_threshold=float(line.strip().split(',')[-1])

    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)

    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),test_pred[:,1],best_threshold)

    with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
        f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
        {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  

    snapshot_inverse_map={v:k for k, v in test_snapshot_map.items()}
    thread_inverse_map={v:k for k, v in test_thread_map.items()}

    fieldnames = ['snapshot_id','thread_id','is_feedback','text_length','True_label', 'Predicted_label', 'Predicted_prob','best_threshold']
    file_name="predictions_"+str(args.val_min_recall).split(".")[-1]+".csv"
    with open(os.path.join(output_dir,file_name),'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k, t, m, n, p, q in zip(test_snapshot_id, test_thread_id, test_feedback, test_text_length,test_target, y_pred, test_pred[:,1], [best_threshold]*len(y_pred)):
            writer.writerow(
                {'snapshot_id':snapshot_inverse_map[i],'thread_id':thread_inverse_map[j],'is_feedback':k, 'text_length': t, 'True_label': m, 
                 'Predicted_label': n, 'Predicted_prob': p, 'best_threshold': q})  


    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))

    print()
    print(f"\n===========Test Set Performance===============\n")
    print()

    print(classification_report(test_target, y_pred))
    print()
    print(confusion_matrix(test_target, y_pred))  
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--multiple_gpus', action="store_true", help="use multiple gpus or not")
    parser.add_argument('--gpus', type=int, default=[], nargs='+', help='used gpu')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    
    parser.add_argument('--customized_model',  action="store_true")
    parser.add_argument('--fp16',  action="store_true")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--validation_split",  type=float,default=0.2,help="ratio to split training and validation set")
    parser.add_argument("--closed_status", action="store_true", help="only keep status=closed") 
    parser.add_argument("--test_date", type=str, default="04_23", help="the month for test set")   
    
    args= parser.parse_args()

    if args.customized_model:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_" + args.model_name.split("-")[2] + "_customized"
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_customized"
    else:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]+ "_" + args.model_name.split("-")[2]
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]
            
    seed_everything(args.seed)

    print()
    print(args)
    print()
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v4_new_email","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))

    if args.closed_status:
        ### only keep emails with status=closed
        df=df[df.state=="closed"]
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    if args.test_date=="04_23":
        ## train: 09/2022 ~ 02/2023. validation: 03/2023  test: 04/2023
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2,3]) else "test"
    elif args.test_date=="03_23":
        df=df[df['time']<'2023-04-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2]) else "test"
    elif args.test_date=="02_23":
        df=df[df['time']<'2023-03-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1]) else "test"        
    else:
        raise ValueError("Invalid test_date. Only 02_23,03_23,04_23 are supported currently")
        
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    df.loc[:,'is_feedback']=df.loc[:,'is_feedback'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    ### validation set use potential complaint :  is_complaint=Y or is_feedback=Y
    ### test set use ground true complaint: is_complaint=Y
    df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']==1),1,0)
    
    df1=df[df.data_type=="train"]
    df1=df1.reset_index(drop=True)
    df_train,df_val=val_mask_creation(df1,'target', validation_split=args.validation_split)
    
    df_test=df[df.data_type=="test"]
    df_test=df_test.reset_index(drop=True)
    ## overwrite the target with the ground true complaint label
    df_test['target']=df_test['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)

    val_df=datasets.Dataset.from_pandas(df_val)
    test_df=datasets.Dataset.from_pandas(df_test)
    hf_data=DatasetDict({"val":val_df,"test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id','thread_id','time','is_feedback',\
                                    'preprocessed_email','is_complaint','target','text_length'])
    
    
    if args.multiple_gpus:
        device=[torch.device(f"cuda:{i}") for i in args.gpus]
    else:
        device=torch.device(args.device)

    val_data=hf_data['val'].shuffle(seed=101).select(range(len(hf_data["val"])))
    val_data.set_format(type="pandas")
    val_data=val_data[:]
    # val_data=val_data.sample(500)
    
    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    test_data.set_format(type="pandas")
    test_data=test_data[:]
    # test_data=test_data.sample(500)
    
    val_unique_snapshot=val_data["snapshot_id"].unique()
    val_snapshot_map={v:idx for idx ,v in enumerate(val_unique_snapshot)}
    val_unique_thread=val_data["thread_id"].unique()
    val_thread_map={v:idx for idx ,v in enumerate(val_unique_thread)}

    val_data["snapshot_id"]=list(map(val_snapshot_map.get,val_data["snapshot_id"]))
    val_data["thread_id"]=list(map(val_thread_map.get,val_data["thread_id"]))

    test_unique_snapshot=test_data["snapshot_id"].unique()
    test_snapshot_map={v:idx for idx ,v in enumerate(test_unique_snapshot)}
    test_unique_thread=test_data["thread_id"].unique()
    test_thread_map={v:idx for idx ,v in enumerate(test_unique_thread)}

    test_data["snapshot_id"]=list(map(test_snapshot_map.get,test_data["snapshot_id"]))
    test_data["thread_id"]=list(map(test_thread_map.get,test_data["thread_id"]))
    
    main(args,val_data, test_data, test_snapshot_map, test_thread_map, device)
    