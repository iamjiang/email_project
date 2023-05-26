import csv
import argparse
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader

from src.mapping import get_peft_config, get_peft_model
from src.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from src.config import PeftType, TaskType, PeftConfig, PromptLearningConfig
from src.peft_model import PeftModel
from src.prefix_tuning import PrefixTuningConfig
from src.prompt_tuning import PromptTuningConfig
from src.p_tuning import PromptEncoderConfig

import evaluate
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    default_data_collator,
    AdamW,
    get_linear_schedule_with_warmup,
    get_scheduler,
    set_seed
)

from accelerate import Accelerator

from tqdm import tqdm

import bitsandbytes as bnb

import utils

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

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
        
        return dict(
            input_ids=_ids,
            attention_mask=_mask,
            labels=_target,
            snapshot_id=_snapshot_id,
            thread_id=_thread_id,
            is_feedback=_is_feedback
        )

    
    def collate_fn(self,batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask=torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        snapshot_id=torch.stack([torch.tensor(x["snapshot_id"]) for x in batch])
        thread_id=torch.stack([torch.tensor(x["thread_id"]) for x in batch])
        is_feedback=torch.stack([torch.tensor(x["is_feedback"]) for x in batch])
        
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
            is_feedback=is_feedback
        )
    

def main(args,  val_data, test_data, test_snapshot_map, test_thread_map, device):
    
    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    
    output_dir=os.path.join(os.getcwd(), args.output_dir)
    
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    peft_model_id = f"{args.model_name}_{peft_config.peft_type}_{peft_config.task_type}"

    config=PeftConfig.from_pretrained(output_dir,peft_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(output_dir,model, peft_model_id)

    if any(k in args.model_name.split("-") for k in ["gpt", "opt", "bloom","GPT"]):
        padding_side = "left"
    else:
        padding_side = "right"

    base_model_config=AutoConfig.from_pretrained(config.base_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, \
                                              padding_side=padding_side,\
                                              model_max_length=args.max_token_length-args.num_virtual_tokens)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    val_module=Loader_Creation(val_df, tokenizer,args.feature_name)
    val_dataloader=DataLoader(val_module,
                              shuffle=False,
                              batch_size=args.batch_size,
                              collate_fn=val_module.collate_fn,
                              pin_memory=True,
                              # num_workers=4
                              )
        
    test_module=Loader_Creation(test_df, tokenizer,args.feature_name)   
    test_dataloader=DataLoader(test_module,
                               shuffle=False,
                               batch_size=args.batch_size,
                               collate_fn=test_module.collate_fn,
                               pin_memory=True,
                               # num_workers=4
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

            batch_idx+=1
            
        return np.concatenate(fin_outputs), np.concatenate(fin_targets), np.concatenate(snapshot_id) , np.concatenate(thread_id) , np.concatenate(is_feedback)
    
    val_pred,val_target,_,_,_=eval_func(val_dataloader,model, device)
    test_pred,test_target,test_snapshot_id,test_thread_id,test_feedback=eval_func(test_dataloader,model, device)
        
    output_dir=os.path.join(os.getcwd(),args.output_dir)

        
    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)

    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),test_pred[:,1],best_threshold)

    with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
        f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
        {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  

    snapshot_inverse_map={v:k for k, v in test_snapshot_map.items()}
    thread_inverse_map={v:k for k, v in test_thread_map.items()}

    fieldnames = ['snapshot_id','thread_id','is_feedback','True_label', 'Predicted_label', 'Predicted_prob','best_threshold']
    file_name="predictions_"+str(args.val_min_recall).split(".")[-1]+".csv"
    with open(os.path.join(output_dir,file_name),'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k, m, n, p, q in zip(test_snapshot_id, test_thread_id, test_feedback, test_target, y_pred, test_pred[:,1], [best_threshold]*len(y_pred)):
            writer.writerow(
                {'snapshot_id':snapshot_inverse_map[i],'thread_id':thread_inverse_map[j],'is_feedback':k, 'True_label': m, 
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
    parser = argparse.ArgumentParser(description='Prompt tuning for language model')
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument('--multiple_gpus', action="store_true", help="use multiple gpus or not")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--model_name', type=str, default = "roberta-large")
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument('--customized_model',  action="store_true")
    parser.add_argument("--peft_type", default="P-tuning", type=str, help="prompt tuning type")
    parser.add_argument("--num_virtual_tokens", default=20, type=int)
    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    
    args,_= parser.parse_known_args()
    
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

    print()
    print(args)
    print()

    seed_everything(args.seed)
    
    if args.peft_type=="P-tuning":
        peft_type = PeftType.P_TUNING
    elif args.peft_type=="Prefix-tuning":
        peft_type = PeftType.PREFIX_TUNING
    elif args.peft_type=="Prompt_Tuning":
        peft_type = PeftType.PROMPT_TUNING
    else:
        raise ValueError("peft_type has to be P-tuning, prefix-tuning or prompt tuning.")

    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v2_new_email","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))

    ### only keep emails with status=closed
    df=df[df.state=="closed"]
    
    ### keep short emails with text_length<=512
    df=df[df.text_length<=512]
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2]) \
    else ("val" if (row["year"]==2023 and row["month"]==3) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    df.loc[:,'target']=df.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    df.loc[:,'is_feedback']=df.loc[:,'is_feedback'].progress_apply(lambda x: 1 if x=="Y" else 0)

    val_df=datasets.Dataset.from_pandas(df[df["data_type"]=="val"])
    test_df=datasets.Dataset.from_pandas(df[df["data_type"]=="test"])
    hf_data=DatasetDict({"val":val_df,"test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id','thread_id','time','is_feedback',\
                                    'preprocessed_email','is_complaint','target'])
    
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
    
