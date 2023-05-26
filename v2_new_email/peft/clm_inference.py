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
    
    
    
class test_loader_creation(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 feature_name,
                 model_max_length
                ):
        super().__init__()
        self.dataset=dataset
        self.tokenizer=tokenizer
        self.feature_name=feature_name
        self.model_max_length=model_max_length
        
        self.dataset=self.dataset.map(self.preprocess_function, 
                                      batched=True,
                                      num_proc=1,
                                      load_from_cache_file=False,
                                      desc="Running tokenizer on dataset",
                                     )

        self.dataset.set_format(type="pandas")
        self.dataset=self.dataset[:]
    
    def preprocess_function(self,examples):
        
        indexed_tokens=self.tokenizer(examples[self.feature_name],add_special_tokens=False).input_ids
        batch_size=len(indexed_tokens)
        prompt=["Email text : ", "Label : "]
        indexed_prompt_v1=self.tokenizer(prompt[0],add_special_tokens=False).input_ids
        indexed_prompt_v2=self.tokenizer(prompt[1],add_special_tokens=False).input_ids
        remaining=self.model_max_length-len(indexed_prompt_v1)-len(indexed_prompt_v2)
            
        inputs = [f"Email text : {self.tokenizer.decode(x[:remaining])} Label : " for x in indexed_tokens]
        model_inputs = self.tokenizer(inputs)
        
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] \
            * (self.model_max_length - len(sample_input_ids)) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.model_max_length - len(sample_input_ids))\
            + model_inputs["attention_mask"][i]
            
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.model_max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.model_max_length])    
        
        return model_inputs

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
    

def main(args, test_data, device):
    
    test_df=Dataset.from_pandas(test_data)
    
    if args.deduped:
        output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project/peft",args.output_dir,"dedup")
    else:
        output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project/peft",args.output_dir)
    
    peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128)
    peft_model_id = f"{args.model_name}_{peft_config.peft_type}_{peft_config.task_type}"

    config=PeftConfig.from_pretrained(output_dir,peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(output_dir,model, peft_model_id)

    if any(k in args.model_name.split("-") for k in ["gpt", "opt", "bloom","GPT"]):
        padding_side = "left"
    else:
        padding_side = "right"

    base_model_config=AutoConfig.from_pretrained(config.base_model_name_or_path)

    if any(k in args.model_name.split("-") for k in ["bloom"]):
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, \
                                              padding_side=padding_side,\
                                              model_max_length=2048-args.num_virtual_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, \
                                                  padding_side=padding_side,\
                                                  model_max_length=base_model_config.max_position_embeddings-2-args.num_virtual_tokens)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.model_max_length:
        model_max_length=tokenizer.model_max_length
    else:
        model_max_length=2048

    test_module=test_loader_creation(test_df, tokenizer,"preprocessed_email",model_max_length)

    test_dataloader=DataLoader(test_module,
                               shuffle=False,
                               batch_size=args.batch_size,
                               collate_fn=test_module.collate_fn,
                               pin_memory=True,
                               # num_workers=4
                               )

    print()
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    print()
    
    model.to(device)
    model.eval()
    
    y_target=[]
    y_pred=[]
    for step, batch in enumerate(tqdm(test_dataloader,position=0 ,leave=True)):
        X=[]
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(input_ids=inputs["input_ids"], \
                                     attention_mask=inputs["attention_mask"], \
                                     max_new_tokens=1, \
                                     eos_token_id=tokenizer.pad_token_id
                                    )
            text_output=tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            
            for result in text_output: 
                if result[0].split(":")[-1].strip()=="complaint":
                    X.extend([1])
                else:
                    X.extend([0])  
                    
        y_target.extend(batch["labels"].cpu().detach().numpy())
        y_pred.extend(X)
        

        if args.deduped:
            output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project/peft",args.output_dir,"dedup")
        else:
            output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project/peft",args.output_dir)

        
        fieldnames = ['True label', 'Predicted label']
        with open(os.path.join(output_dir,"predictions.csv"),'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for i, j in zip(y_target, y_pred):
                writer.writerow(
                    {'True label': i, 'Predicted label': j})  

#         test_output=utils.model_evaluate(y_target,y_pred)

#         print("==> performance on test set \n")
#         print("")
#         print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
#                      test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))

#         print()
#         print(f"\n===========Test Set Performance===============\n")
#         print()

#         print(classification_report(y_target, y_pred))
#         print()
#         print(confusion_matrix(y_target, y_pred))          
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prompt tuning for language model')
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--model_name', type=str, default = "roberta-large")
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--peft_type", default="P-tuning", type=str, help="prompt tuning type")
    parser.add_argument('--deduped', action="store_true", help="keep most recent thread id or not")
    parser.add_argument("--num_virtual_tokens", default=20, type=int)
    parser.add_argument('--text_label', nargs='+', default=[" no complaint"," complaint"])
    
    args,_= parser.parse_known_args()
    
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

    if args.deduped:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_dedup_data")
    else:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_data")

    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))

    df['preprocessed_email']=df['preprocessed_email'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1]) \
    else ("val" if (row["year"]==2023 and row["month"]==2) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)

    df.loc[:,'target']=df.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)

    test_df=datasets.Dataset.from_pandas(df[df["data_type"]=="test"])
    hf_data=DatasetDict({"test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id','thread_id','time','is_feedback',\
                                    'preprocessed_email','is_complaint','target'])

    # classes=[" no complaint"," complaint"]
    classes=args.text_label
    hf_data = hf_data.map(
        lambda x: {"text_label": [classes[label] for label in x["target"]]},batched=True,num_proc=1)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    test_data.set_format(type="pandas")
    test_data=test_data[:]
    
    main(args,test_data, device)
    
