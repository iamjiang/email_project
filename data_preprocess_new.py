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

from fuzzywuzzy import fuzz

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
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
           "Good Evening",
           "do not click on the links or attachments unless you recognize the sender and you know the content is safe.",
           "If you have received this message in error, please advise the sender by reply email and delete the message.",
           "The information contained in this email message is considered confidential and proprietary to the sender and is intended solely for review and use by the named recipient",
           "Any unauthorized review, use or distribution is strictly prohibited",
           "If you are not the intended recipient, please delete this message and notify the sender immediately.",
           "if you are not the intended recipient, please contact the sender by reply email and destroy all copies of the original message.",
           "if you are not the intended recipient, you are hereby notified that any review, dissemination, distribution, or duplication of this communication is strictly prohibited.",
           "if you received this transmission in error, please immediately contact the sender and destroy the material in its entirety, whether in electronic or hard copy",
           
           "if you are not the intended recipient, you are hereby notified that any disclosure, copying, distribution, or use of the prohibited.",
           "if you have received this message in error, please send it back to us, and immediately and permanently delete it.",
           "it is intended only for the use of the persons it is addressed to.",
           "the information contained in this transmission may contain privileged and confidential information.",
           "your privacy is important to us.",
           "do not use, copy or disclose the information contained in this message or in any attachment.",
           "if you have questions or need assistance or give me a call",
           "we aim to exceed your expectations.",
           "please feel free to let me know if you have further questions.",
           "please let me know if you have any questions or concerns.",
           "tell us how we are doing better together.",
           "tell us how we are doing better subject re fwd genuine cable group, llc morgan testing following up.",
           "please do not reply to this email address.",
           "this is a system generated message.",
           "this alert was sent according to your settings.",
           "subject re revolver increase - amendment no.",
           "i hope you are well.",
           "i hope your day is going by well.",
           "if there are questions / comments, pls let me know.",
           "thank you.",
           "Thank you for getting back to me.",
           "happy friday.",
           "disclaimer this email and any attachments are confidential and for the sole use of the recipients.",
           "if necessary for your business with jpmc, please save the decrypted content of this email in a secure location for future reference.",
           "copyright 2015 jpmorgan chase co. all rights reserved",
           "this email has been scanned for viruses and malware, and may have been automatically archived by mimecast ltd, an innovator in software as a service saas for business.",
           "providing a safer and more useful place for your human generated data.",
           "specializing in; security, archiving and compliance.",
           "to find out more click here.",
           "hi all.",
           "i appreciate the patience, let me know if you have additional questions.",
           "This and any attachments are intended solely for the or to whom they are addressed",
           "use of the information in this may be a of the law",
           
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

def remove_similar_phrase(text):
    sent_list=[]
    sent_text = nltk.sent_tokenize(text)
    
    threshold=90
    
    for p in phrases:
        for sent in sent_text:
            score=fuzz.token_set_ratio(sent, p.lower())
            if score>=threshold:
                sent_text.remove(sent)
    sent_list=[str(v).strip().strip("\n") for v in sent_text]
    return " ".join(sent_list)

## remove all non-English Word
# def remove_non_english(text):
#     snow_stemmer = SnowballStemmer(language='english')
#     words = set(nltk.corpus.words.words())
#     text=" ".join(w for w in nltk.wordpunct_tokenize(text) if snow_stemmer.stem(w).lower() in words or not w.isalpha())
#     return text

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

phrase=["This and any attachments are intended solely",
        "If you have received this message in error",
        "If you have received this communication in error",
        "If you have received this e-mail and are not an intended recipient",
        "If you received this fax in error",
        "If the reader of this message is not the prohibited",
        "If the reader of this message is not the intended recipient",
        "If you no longer wish to receive these emails",
        "This message, and any attachments to it, may disclosure under applicable law",
        "attachments should be read and retained by intended",
        "Please feel free to reach out to me",
        "This e-mail, including any attachments that accompany it",
        "It is intended exclusively for the individual or entity",
        "This communication may contain information that is proprietary",
        "If you are not the named addressee",
        "Please keep me posted if I can help in any way.",
        "Have a great weekend"]
def remove_certain_phrase(text):
    sent_list=[]
    sent_text=nltk.sent_tokenize(text)
    
    for p in phrase:
        regex=re.compile(p,re.IGNORECASE)
        for sent in sent_text:
            if regex.search(sent):
                sent_text.remove(sent)
        
    return " ".join(sent_text)

df['preprocessed_email'] = df['preprocessed_email'].astype(str)

# df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_appos)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_layout)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_similar_phrase)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_certain_phrase)
# df['preprocessed_email'] = df['preprocessed_email'].progress_apply(remove_non_english)
df['preprocessed_email'] = df['preprocessed_email'].progress_apply(clean_re)

df.dropna(subset=['preprocessed_email'],inplace=True)
df=df[df.preprocessed_email.notna()]
df=df[df.preprocessed_email.str.len()>0]

input_dir=os.path.join(os.getcwd(),"datasets")
df.to_pickle(os.path.join(input_dir,"new_train_val_test_pickle"))

