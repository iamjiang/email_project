import argparse
import pandas as pd
pd.options.mode.chained_assignment=None #default="warn"
import numpy as np
from numpy import savez_compressed, load
import itertools
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

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
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
# Load the stopwords from the new directory
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
nltk.data.path.append(nltk_data_dir)

import spacy
model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","en_core_web_md","en_core_web_md-3.3.0")
nlp = spacy.load(model_name)

from transformers import AutoTokenizer
from datasets import Dataset

#NER: remove lines that contain only names, phone numbers, addresses, attachments e.g. [cid:image001.jpg@01D84D20.F7040E40], JPMC

#     end_greetings_exact_match = ['^thank you[\s]*[.!,]?$', '^thanks[\s]*[.!,]?$', 
#                                  "^thank you in advance[\s]*[.!,]$", 
#                                  "^thanks in advance[\s]*[.!,]$",
#                                 "Good Morning","Good Afternoon","Good Evening"]

#     end_greetings = ['regards', 'best regards', 'best,', 'thanks & regards', 
#                      'sent with blackberry', 'thanks and regards', 'with thanks']


jpmc_statutory_msgs = \
['this message may include proprietary or protected information',
 'This message, and any attachments to it, may disclosure under applicable law',
 'action required - ',
 'please feel free to let me know if you have further questions.',
 'disclaimer',
 'the information contained in this transmission may contain privileged and confidential information.',
 'tell us how we are doing better together.',
 'notice:',
 'Have a great weekend',
 'if you have received this message in error, please send it back to us, and immediately and permanently delete it.',
 'Please keep me posted if I can help in any way.',
 'do not click on the links or attachments unless you recognize the sender and you know the content is safe.',
 'received: from icprdc',
 'if you have questions or need assistance or give me a call',
 'please let me know if you have any questions or concerns.',
 'If you no longer wish to receive these emails',
 'hi all.',
 'this email has been scanned for viruses',
 'This and any attachments are intended solely',
 'thank you.',
 'The information contained in this email message is considered confidential and proprietary to the sender and is intended solely for review and use by the named recipient',
 'My work hours may not be yours. Please do not feel obligated to respond outside of your normal work hours.',
 'tell us how we are doing',
 'this message is confidential',
 'caution: this email',
 'Any unauthorized review, use or distribution is strictly prohibited',
 'the information in this email and any attachments',
 'disclaimer this email and any attachments are confidential and for the sole use of the recipients.',
 'i hope you are well.',
 'use of the information in this may be a of the law',
 'the information contained in this communication',
 'i hope your day is going by well.',
 'If you received this fax in error',
 'important reminder: j.p.',
 'This communication may contain information that is proprietary',
 'If you have received this communication in error',
 'if you are not the intended recipient, please contact the sender by reply email and destroy all copies of the original message.',
 'do not use, copy or disclose the information contained in this message or in any attachment.',
 'this electronic transmission is',
 'this transmission may',
 'If you have received this message in error, please advise the sender by reply email and delete the message.',
 'attachments should be read and retained by intended',
 'Please feel free to reach out to me',
 'providing a safer and more useful place for your human generated data.',
 'If you are not the named addressee',
 'If you are not the intended recipient, please delete this message and notify the sender immediately.',
 'This and any attachments are intended solely for the or to whom they are addressed',
 'If you have received this e-mail and are not an intended recipient',
 'copyright 2015 jpmorgan chase co. all rights reserved',
 'your privacy is important to us.',
 'This e-mail, including any attachments that accompany it',
 'this is a system generated message.',
 'confidentiality notice: This e-mail and any documents',
 'if you received this transmission in error, please immediately contact the sender and destroy the material in its entirety, whether in electronic or hard copy',
 'If the reader of this message is not the prohibited',
 'we aim to exceed your expectations.',
 'i appreciate the patience, let me know if you have additional questions.',
 'happy friday.',
 'jpmorgan chase sent you a document to review and sign',
 'if necessary for your business with jpmc, please save the decrypted content of this email in a secure location for future reference.',
 'if you are not the intended recipient, you are hereby notified that any review, dissemination, distribution, or duplication of this communication is strictly prohibited.',
 'if there are questions / comments, pls let me know.',
 'the information contained in this transmission is confidential',
 'Thank you for getting back to me.',
 'received: from vsinthank you for choosing jpmorgan chase for your',
 'subject re revolver increase - amendment no.',
 'the information contained in this e-mail and any accompanying documents',
 'It is intended exclusively for the individual or entity',
 'this email is for the use of',
 'to find out more click here.',
 'view additional information on',
 'it is intended only for the use of the persons it is addressed to.',
 'this email has been scanned for viruses and malware, and may have been automatically archived by mimecast ltd, an innovator in software as a service saas for business.',
 'specializing in; security, archiving and compliance.',
 'this alert was sent according to your settings.',
 'If you have received this message in error',
 'this is a secure',
 'please do not reply to this email address.',
 'unless expressly stated',
 'all rights reserved',
 'If the reader of this message is not the intended recipient',
 'if you are not the intended recipient, you are hereby notified that any disclosure, copying, distribution, or use of the prohibited.',
'Any unauthorized use is strictly prohibited',
'It is intended solely for use by the recipient',
 'You are hereby notified that any disclosure, copying, distribution',
]


# jpmc_statutory_msgs=jpmc_statutory_msgs+end_greetings_exact_match+end_greetings

remove_layout_starting_with = ['from:', 'date:', 'sent:', 'to:', 'cc:',\
                               'subject:', 'importance:', "reply-to:",\
                              'mailto:',]

address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
url_pattern = "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

us_phone_num_pattern = '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'
email_id_pattern = "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"  

def clean_line(line):
    if line.startswith(">"):
        line = line[1:]
    return line.strip()

def remove_layout(text):
    text = re.sub('_x000D_', "", text)
    text = re.sub("\r", "", text)
    text = "\n".join([clean_line(line) for line in text.split("\n")])
    paragraphs = text.split("\n\n")
    # paragraphs.remove("")

    filtered_paragraph=[]

    for paragraph in paragraphs:
        keep_paragraph=True

        for pattern in remove_layout_starting_with:
            if re.search(pattern, paragraph.lower()):
                keep_paragraph=False
                break

        ## remove signature containing address
        if re.search(address_pattern, paragraph.lower()):
            keep_paragraph=False
        ## remove signature containing phone_num
        if re.search(us_phone_num_pattern, paragraph.lower()):
            keep_paragraph=False

        if keep_paragraph:
            filtered_paragraph.append(paragraph)
    # remove layout information (short-text split by '\n\n')           
    filter_text=[i for i in filtered_paragraph if len(i.split())>=10]                 
    return "\n\n".join(filter_text)

def remove_duplicates_freq(text):
    paragraphs=text.split("\n\n")
    unique_paragraphs=[]
    duplicated_paragraphs=set()
    for p in paragraphs:
        if p not in duplicated_paragraphs:
            unique_paragraphs.append(p)
            duplicated_paragraphs.add(p)
            
    return "\n\n".join(unique_paragraphs)

def similarity(text1,text2):
    doc1=nlp(text1)
    doc2=nlp(text2)
    return cosine_similarity(doc1.vector.reshape(1,-1),doc2.vector.reshape(1,-1))[0][0]

### semantic search to remove duplicated paragraphs
def remove_duplicates_semantic(text):
    paragraphs = text.split("\n\n")
    similarities={}
    for i,j in combinations(range(len(paragraphs)),2):
        similarity_score=similarity(paragraphs[i],paragraphs[j])
        similarities[(i,j)]=similarity_score
        similarities[(j,i)]=similarity_score
    unique_paragraphs=[]
    for i in range(len(paragraphs)):
        if i not in [p[0] for p in unique_paragraphs]:
            unique=True
            for j in range(i+1,len(paragraphs)):
                if j not in [p[0] for p in unique_paragraphs] and similarities[(i,j)]>0.98:
                    unique=False
                    break
            if unique:
                unique_paragraphs.append((i,paragraphs[i]))
    
    unique_paragraphs.sort(key=lambda p : str(text).index(p[1]))
    return "\n\n".join([p[1] for p in unique_paragraphs])

def clean_re(text):

    text = re.sub(r"[^A-Za-z\s\d+\/\d+\/\d+\.\:\-,;?'\"%$]", '', text) # replace non-alphanumeric with space
    #remove long text such as encrypted string
    text = " ".join(word for word in text.split() if len(word)<=20) 
    
    #remove non-alphanumerc characters: '\u200c\xa0\u200c\xa0\u200c\xa0\n'
    text = re.sub(r"\u200c\xa0+",'',text)
    
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{1,}", " ", text)
    text = re.sub(r"\t{1,}", " ", text)
    text = re.sub("_{2,}","",text)
    text = re.sub("\[\]{1,}","",text)
    text = re.sub(r"(\s\.){2,}", "",text) #convert pattern really. . . . . . .  gotcha  into really. gotcha 

    # Define regular expression pattern for address and signature
    address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
    # signature_pattern = r"^\s*[a-zA-Z0-9\s,]+\s*$"

    url_pattern = \
    "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

    us_phone_num_pattern = \
    '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'

    email_id_pattern = \
    "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    # Remove address and signature from email
    text = re.sub(address_pattern, "", text)
    # text = re.sub(signature_pattern, "", text)
    text = re.sub(url_pattern, "", text)
    text = re.sub(us_phone_num_pattern, "", text)
    text = re.sub(email_id_pattern, "", text)

    return text

def fuzz_search_phrase(text):
    sent_list=[]
    sent_text = nltk.sent_tokenize(text)

    threshold=90

    for p in jpmc_statutory_msgs:
        for sent in sent_text:
            score=fuzz.token_set_ratio(sent, p.lower())
            if score>=threshold:
                sent_text.remove(sent)
    sent_list=[str(v).strip().strip("\n") for v in sent_text]
    return " ".join(sent_list)

def further_remove_layout(text):
    # remove layout information (short-text)
    text = re.sub(r"\n{1,}", "\n", text)
    x=[i for i in text.split("\n") if len(i.split())>=10]  
    return "\n".join(x)

def remove_certain_phrase(text):
    sent_list=[]
    sent_text=nltk.sent_tokenize(text)

    for p in jpmc_statutory_msgs:
        regex=re.compile(p,re.IGNORECASE)
        for sent in sent_text:
            if regex.search(sent):
                sent_text.remove(sent)

    return " ".join(sent_text)

def main(df, args,output_dir):
    
    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    print()
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print()

    df.dropna(subset=['email'],inplace=True)
    df=df[df.email.notna()]
    df=df[df.email.str.len()>0]
    
    df['email'] = df['email'].astype(str)
    df['email'] = df['email'].progress_apply(remove_layout)
    df['email'] = df['email'].progress_apply(remove_duplicates_freq)
    # df['email'] = df['email'].progress_apply(remove_duplicates_semantic)

    # df['email'] = df['email'].progress_apply(further_remove_layout)
    
    df['email'] = df['email'].progress_apply(clean_re)
    df['email'] = df['email'].progress_apply(remove_certain_phrase)
    df['email'] = df['email'].progress_apply(fuzz_search_phrase)
    df['preprocessed_email'] = df['email'].progress_apply(clean_re)

    df.dropna(subset=['preprocessed_email'],inplace=True)
    df=df[df.preprocessed_email.notna()]
    df=df[df.preprocessed_email.str.len()>0]
        
    hf_data=Dataset.from_pandas(df)
    def compute_lenth(example):
        return {"text_length":len(example["input_ids"])}

    hf_data=hf_data.map(lambda x: tokenizer(x["email"]),batched=True)
    hf_data=hf_data.map(compute_lenth)
    hf_data.set_format(type="pandas")
    df=hf_data[:]
    
    keep_columns=['snapshot_id', 'email', 'gcid', 'thread_id', 'state', 'time','is_complaint', \
              'is_feedback', 'supervised_groups', 'year', 'month','day', 'preprocessed_email','text_length']
    df=df.loc[:,keep_columns]

    df.to_pickle(os.path.join(output_dir,f"train_val_test_pickle_{args.data_index}"))        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='email data preprocessing')
    parser.add_argument('--model_name', type=str, default = "longformer-large-4096")
    parser.add_argument('--data_index',type=int,default=0)
    args= parser.parse_args()
    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter/v2_new_email/datasets/split_data/"
    
    file_name=f"email_data_{args.data_index}.csv"
    df=pd.read_csv(os.path.join(root_dir,file_name))
    print()
    print(df.shape)
    print()

    output_dir=os.path.join(os.getcwd(), "split_data")

    grouped_df=df.groupby('thread_id')
    sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) for _, group in grouped_df]
    df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first").reset_index(drop=True)
        
    main(df, args,output_dir)
    


   



