import argparse
import pandas as pd
import numpy as np
from numpy import savez_compressed, load
import itertools
import re
import time
import os
import pickle

import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

import transformers

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
)
print("Transformers version is {}".format(transformers.__version__))

pd.set_option('display.max_columns', None,'display.max_rows',None)

root_dir="/opt/omniai/work/instance1/jupyter"
data_dir=os.path.join(root_dir,"email-complaints","datasets")
train_df=pd.read_csv(os.path.join(data_dir,"train.csv"))
val_df=pd.read_csv(os.path.join(data_dir,"val.csv"))
test_df=pd.read_csv(os.path.join(data_dir,"test.csv"))

train_df['preprocessed_email'] = train_df['preprocessed_email'].astype(str)
val_df['preprocessed_email'] = val_df['preprocessed_email'].astype(str)
test_df['preprocessed_email'] = test_df['preprocessed_email'].astype(str)

train_df['target']=train_df["is_complaint"].map(lambda x: 1 if x=="Y" else 0)
val_df['target']=val_df["is_complaint"].map(lambda x: 1 if x=="Y" else 0)
test_df['target']=test_df["is_complaint"].map(lambda x: 1 if x=="Y" else 0)

assert train_df[train_df.target==1].shape[0]==train_df[train_df.is_complaint=="Y"].shape[0] 
assert train_df[train_df.target==0].shape[0]==train_df[train_df.is_complaint=="N"].shape[0] 
assert val_df[val_df.target==1].shape[0]==val_df[val_df.is_complaint=="Y"].shape[0] 
assert val_df[val_df.target==0].shape[0]==val_df[val_df.is_complaint=="N"].shape[0] 
assert test_df[test_df.target==1].shape[0]==test_df[test_df.is_complaint=="Y"].shape[0] 
assert test_df[test_df.target==0].shape[0]==test_df[test_df.is_complaint=="N"].shape[0] 

hf_train=Dataset.from_pandas(train_df)
hf_val=Dataset.from_pandas(val_df)
hf_test=Dataset.from_pandas(test_df)

hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})
hf_data=hf_data.filter(lambda x: x['preprocessed_email']!=None)

model_dir=os.path.join(root_dir,"transformers-models")
model_checkpoint=os.path.join(model_dir,"longformer-large-4096")
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

train_df1=Dataset.from_pandas(train_df)
train_df1=train_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

val_df1=Dataset.from_pandas(val_df)
val_df1=val_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

test_df1=Dataset.from_pandas(test_df)
test_df1=test_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

def compute_lenth(example):
    return {"text_length":len(example["input_ids"])}
train_df1=train_df1.map(compute_lenth)
val_df1=val_df1.map(compute_lenth)
test_df1=test_df1.map(compute_lenth)

# train_df1=train_df1.filter(lambda x: x['text_length']>10)
# val_df1=val_df1.filter(lambda x: x['text_length']>10)
# test_df1=test_df1.filter(lambda x: x['text_length']>10)

train_df1.save_to_disk(os.path.join(data_dir,'train_df'))
val_df1.save_to_disk(os.path.join(data_dir,'val_df'))
test_df1.save_to_disk(os.path.join(data_dir,'test_df'))
             

