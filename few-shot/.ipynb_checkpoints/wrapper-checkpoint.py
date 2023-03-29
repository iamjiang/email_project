import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()

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
        indexed_prompt=tokenizer(prompt[0], return_tensors="pt",add_special_tokens=False).input_ids
        remaining=tokenizer.model_max_length-indexed_prompt.shape[1]-2
        df["new_input_ids"]=df["input_ids"].progress_apply(lambda x: np.concatenate([x[:remaining],indexed_prompt.squeeze()]))
        df["wrapped_email"]=df["new_input_ids"].progress_apply(lambda x: tokenizer.decode(x))
    elif len(prompt)>1:
        indexed_prompt_v1=tokenizer(prompt[0], return_tensors="pt",add_special_tokens=False).input_ids
        indexed_prompt_v2=tokenizer(prompt[1], return_tensors="pt",add_special_tokens=False).input_ids
        remaining=tokenizer.model_max_length-indexed_prompt_v1.shape[1]-indexed_prompt_v2.shape[1]-2
        df["new_input_ids"]=df["input_ids"].progress_apply(lambda x: np.concatenate([indexed_prompt_v1.squeeze(),x[:remaining],indexed_prompt_v2.squeeze()]))
        df["wrapped_email"]=df["new_input_ids"].progress_apply(lambda x: tokenizer.decode(x))    
        
    df['target']=df["is_complaint"].map(lambda x: 1 if x=="Y" else 0)
    
    wrapped_text=datasets.Dataset.from_pandas(df)
    
    wrapped_text=wrapped_text.select_columns(['wrapped_email','target'])
    
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
    
    