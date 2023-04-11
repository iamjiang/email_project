import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()

import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict
from datasets import load_from_disk


def prompt_wrapper(text, prompt, tokenizer):

    text=text.map(lambda x:tokenizer(x['preprocessed_email'],add_special_tokens=False), batched=True).remove_columns("attention_mask")
    def compute_lenth(example):
        return {"text_length":len(example["input_ids"])}
    text=text.map(compute_lenth)
    text=text.sort('text_length')
    
    text.set_format(type="pandas")
    df=text[:]
    
    df.dropna(subset=['preprocessed_email'],inplace=True)
    df=df[df['preprocessed_email']!="nan"]
    
    if len(prompt)==1:
        def decode_tokens_func(example):
            indexed_prompt=tokenizer(prompt[0], add_special_tokens=False).input_ids
            remaining=tokenizer.model_max_length-len(indexed_prompt)-2-1 # 2 for special token, 1 for addtional space between the last token and prompt
            decoded_x0=tokenizer.decode(example[:remaining], skip_special_tokens=True)
            decoded_x1=tokenizer.decode(indexed_prompt, skip_special_tokens=False)
            result=decoded_x0+". "+decoded_x1
            # result=decoded_x0+"."+decoded_x1.replace("<mask>"," <mask>")
            new_input_ids=tokenizer(result,add_special_tokens=False).input_ids
            return new_input_ids
        df["new_input_ids"]=df["input_ids"].progress_apply(decode_tokens_func)
        df["wrapped_email"]=df["new_input_ids"].progress_apply(lambda x: tokenizer.decode(x,skip_special_tokens=False))

    elif len(prompt)>1:
        indexed_prompt_v1=tokenizer(prompt[0], return_tensors="pt",add_special_tokens=False).input_ids
        indexed_prompt_v2=tokenizer(prompt[1], return_tensors="pt",add_special_tokens=False).input_ids
        remaining=tokenizer.model_max_length-indexed_prompt_v1.shape[1]-indexed_prompt_v2.shape[1]-2
        df["new_input_ids"]=df["input_ids"].progress_apply(lambda x: np.concatenate([indexed_prompt_v1.squeeze(),x[:remaining],indexed_prompt_v2.squeeze()]))
        df["wrapped_email"]=df["new_input_ids"].progress_apply(lambda x: tokenizer.decode(x))    
        
    df['target']=df["is_complaint"].map(lambda x: 1 if x=="Y" else 0)
    
    wrapped_text=datasets.Dataset.from_pandas(df)
    
    wrapped_text=wrapped_text.select_columns(['snapshot_id','preprocessed_email','wrapped_email','is_complaint','target'])
    
    return wrapped_text
        
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
    
    