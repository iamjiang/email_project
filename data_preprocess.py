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

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Load the stopwords from the new directory
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
nltk.data.path.append(nltk_data_dir)

import spacy
model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","en_core_web_md","en_core_web_md-3.3.0")
nlp = spacy.load(model_name)

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


# input_dir=os.path.join(os.getcwd(),"datasets")
# train_df=pd.read_csv(os.path.join(input_dir,"train.csv"))
# val_df=pd.read_csv(os.path.join(input_dir,"val.csv"))
# test_df=pd.read_csv(os.path.join(input_dir,"test.csv"))

# train_df.to_pickle(os.path.join(input_dir,"train_pickle"))
# val_df.to_pickle(os.path.join(input_dir,"val_pickle"))
# test_df.to_pickle(os.path.join(input_dir,"test_pickle"))

input_dir=os.path.join(os.getcwd(),"datasets")

train_df=pd.read_pickle(os.path.join(input_dir,"train_pickle"))
val_df=pd.read_pickle(os.path.join(input_dir,"val_pickle"))
test_df=pd.read_pickle(os.path.join(input_dir,"test_pickle"))

train_df["data_type"]=["training_set"]*len(train_df)
val_df["data_type"]=["validation_set"]*len(val_df)
test_df["data_type"]=["test_set"]*len(test_df)
df=pd.concat([train_df,val_df,test_df],axis=0)

# remove phrases
phrases = ["My work hours may not be yours. Please do not feel obligated to respond outside of your normal work hours.",
           "Good Morning",
           "Good Afternoon",
           "Good Evening"
          ]

for p in phrases:
    df['preprocessed_email']= df['preprocessed_email'].str.replace(p, ' ')
    df['preprocessed_email'] = df['preprocessed_email'].str.replace("`", "'")
    df['preprocessed_email'] = df['preprocessed_email'].str.replace("’", "'")   

appos = {
    "aren’t" : "are not",
    "can’t" : "cannot",
    "couldn’t" : "could not",
    "didn’t" : "did not",
    "doesn’t" : "does not",
    "don’t" : "do not",
    "hadn’t" : "had not",
    "hasn’t" : "has not",
    "haven’t" : "have not",
    "he’d" : "he would",
    "he’ll" : "he will",
    "he’s" : "he is",
    "i’d" : "i would",
    "i’d" : "i had",
    "i’ll" : "i will",
    "i’m" : "i am",
    "isn’t" : "is not",
    "it’s" : "it is",
    "it’ll":"it will",
    "i’ve" : "i have",
    "let’s" : "let us",
    "mightn’t" : "might not",
    "mustn’t" : "must not",
    "shan’t" : "shall not",
    "she’d" : "she would",
    "she’ll" : "she will",
    "she’s" : "she is",
    "shouldn’t" : "should not",
    "that’s" : "that is",
    "there’s" : "there is",
    "they’d" : "they would",
    "they’ll" : "they will",
    "they’re" : "they are",
    "they’ve" : "they have",
    "we’d" : "we would",
    "we’re" : "we are",
    "we’ll" : "we will",
    "weren’t" : "were not",
    "we’ve" : "we have",
    "what’ll" : "what will",
    "what’re" : "what are",
    "what’s" : "what is",
    "what’ve" : "what have",
    "where’s" : "where is",
    "who’d" : "who would",
    "who’ll" : "who will",
    "who’re" : "who are",
    "who’s" : "who is",
    "who’ve" : "who have",
    "won’t" : "will not",
    "wouldn’t" : "would not",
    "you’d" : "you would",
    "you’ll" : "you will",
    "you’re" : "you are",
    "you’ve" : "you have",
    "’re": " are",
    "wasn’t": "was not",
    "didn’t": "did not"
    }

def remove_appos(text):
    text = [appos[word] if word in appos else word for word in text.lower().split()]
    text = " ".join(text)
    return text

def remove_layout(text):
    x=[i for i in text.split("\n") if len(i.split())>=10] # remove layout information (short-text)
    return "\n".join(x)

def remove_short_sentence(text):
    sent_list=[]
    sent_text = nltk.sent_tokenize(text)
    for sent in sent_text:
        # sent=sent.strip().strip("\n")
        # sent=str(sent).strip().strip("\n")
        if len(sent)>5:
            sent_list.append(sent)
    sent_list=[str(v).strip().strip("\n") for v in sent_list]
    return " ".join(sent_list)

def clean_re(text):
   
    text = re.sub(r"[^A-Za-z\s\d+\/\d+\/\d+\.\-,;?'\"%$]", '', text) # replace non-alphanumeric with space
    text = " ".join(word for word in text.split() if len(word)<=20) #remove long text
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{1,}", " ", text)
    text = re.sub(r"\t{1,}", " ", text)
    text = re.sub("_{2,}","",text)
    text = re.sub("\[\]{1,}","",text)
    text = re.sub(r"(\s\.){2,}", "",text) #convert pattern really. . . . . . .  gotcha  into really. gotcha 
    
    # Define regular expression pattern for address and signature
    address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
    signature_pattern = r"^\s*[a-zA-Z0-9\s,]+\s*$"
    # Remove address and signature from email
    text = re.sub(address_pattern, "", text)
    text = re.sub(signature_pattern, "", text)
    
    return text

df['preprocessed_email'] = df['preprocessed_email'].astype(str)

# df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_appos)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_layout)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_short_sentence)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(clean_re)

df.dropna(subset=['preprocessed_email'],inplace=True)
df=df[df.preprocessed_email.notna()]
df=df[df.preprocessed_email.str.len()>0]

input_dir=os.path.join(os.getcwd(),"datasets")
df.to_pickle(os.path.join(input_dir,"train_val_test_pickle"))

