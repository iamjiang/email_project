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

input_dir="./datasets"
train_df=pd.read_csv(os.path.join(input_dir,"train.csv"))
val_df=pd.read_csv(os.path.join(input_dir,"val.csv"))
test_df=pd.read_csv(os.path.join(input_dir,"test.csv"))

train_df['preprocessed_email'] = train_df['preprocessed_email'].astype(str)
val_df['preprocessed_email'] = val_df['preprocessed_email'].astype(str)
test_df['preprocessed_email'] = test_df['preprocessed_email'].astype(str)

hf_train=Dataset.from_pandas(train_df)
hf_val=Dataset.from_pandas(val_df)
hf_test=Dataset.from_pandas(test_df)

hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})

model_checkpoint=os.path.join(os.getcwd(), "transformers-models","longformer-base-4096")
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

train_df1.save_to_disk('train_df')
val_df1.save_to_disk('val_df')
test_df1.save_to_disk('test_df')

