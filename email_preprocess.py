#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !mkdir datasets misc transformers-models


# In[ ]:


# !aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/smarsh-emails/combined/preprocessed/model_training/splits  ./datasets \
# --recursive --exclude "*" --include "*.csv" --exclude "*/*"


# In[ ]:


# !aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/R680994/notebooks/complaints_larger_instance/data/  ./misc/data/  \
# --recursive --exclude "*" --include "*.csv" --include "*.csv.zip" --exclude "*/*"


# In[ ]:


# !aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/R680994/notebooks/complaints_larger_instance/  ./misc/  \
# --recursive --exclude "*" --include "*.ipynb" --include "*.py" --exclude "*/*"


# In[16]:


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

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from IPython.display import display, HTML

sns.set(style="whitegrid",palette='muted',font_scale=1.2)
rcParams['figure.figsize']=16,10

get_ipython().run_line_magic('config', 'InlineBackend.figure_format="retina"')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None,'display.max_rows',None)


# In[2]:


input_dir="./datasets"
train_df=pd.read_csv(os.path.join(input_dir,"train.csv"))
val_df=pd.read_csv(os.path.join(input_dir,"val.csv"))
test_df=pd.read_csv(os.path.join(input_dir,"test.csv"))


# In[3]:


train_df.head()


# In[4]:


hf_train=Dataset.from_pandas(train_df)
hf_val=Dataset.from_pandas(val_df)
hf_test=Dataset.from_pandas(test_df)

hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})


# In[5]:


hf_data


# In[6]:


def label_distribution(df):
    tempt1=pd.DataFrame(df["is_complaint"].value_counts(dropna=False)).reset_index().rename(columns={'index':'is_complaint','is_complaint':'count'})
    tempt2=pd.DataFrame(df["is_complaint"].value_counts(dropna=False,normalize=True)).reset_index().rename(columns={'index':'is_complaint','is_complaint':'percentage'})
    return tempt1.merge(tempt2, on="is_complaint", how="inner")

def style_format(df,  data_type="Training set"):
    return df.style.format({'count':'{:,}','percentage':'{:.2%}'})\
           .set_caption(f"{data_type} label distribution")\
           .set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}])


# In[7]:


label_train=label_distribution(train_df)
style_format(label_train,  data_type="Training set")


# In[8]:


label_val=label_distribution(val_df)
style_format(label_val,  data_type="validation set")


# In[10]:


label_test=label_distribution(test_df)
style_format(label_test,  data_type="Test set")


# In[9]:


model_checkpoint=os.path.join(os.getcwd(), "transformers-models","roberta-base")
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)


# In[11]:


### run the following codes in .py file because the progressive bar is not shown correctly in notebook

# train_df1=Dataset.from_pandas(train_df)
# train_df1=train_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

# val_df1=Dataset.from_pandas(val_df)
# val_df1=val_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

# test_df1=Dataset.from_pandas(test_df)
# test_df1=test_df1.map(lambda x: tokenizer(x["preprocessed_email"]),batched=True)

# def compute_lenth(example):
#     return {"text_length":len(example["input_ids"])}
# train_df1=train_df1.map(compute_lenth)
# val_df1=val_df1.map(compute_lenth)
# test_df1=test_df1.map(compute_lenth)

# train_df1.save_to_disk('train_df')
# val_df1.save_to_disk('val_df')
# test_df1.save_to_disk('test_df')


# In[17]:


def statistics_compute(hf_df1,hf_df2,hf_df3,p=1):

    X=[]
    X.append(np.percentile(hf_df1['text_length'],p))
    X.append(np.percentile(hf_df2['text_length'],p))
    X.append(np.percentile(hf_df3['text_length'],p))
    
    result={}
    result['percentile']=X
    result["min"]=[np.min(hf_df1['text_length']),np.min(hf_df2['text_length']),np.min(hf_df3['text_length'])]
    result["max"]=[np.max(hf_df1['text_length']),np.max(hf_df2['text_length']),np.max(hf_df3['text_length'])]
    result["mean"]=[np.mean(hf_df1['text_length']),np.mean(hf_df2['text_length']),np.mean(hf_df3['text_length'])]
    return result

def statistics_table(hf_df1,hf_df2,hf_df3):
    dict_data={}
    dict_data["data_type"]=["training", "validation", "test"]
    dict_data["# of obs"]=[len(hf_df1['text_length']),len(hf_df2['text_length']),len(hf_df3['text_length'])]
    dict_data["Min of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3)["min"]
    dict_data["1% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=1)['percentile']
    dict_data["5% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=5)['percentile']
    dict_data["10% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=10)['percentile']
    dict_data["25% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=25)['percentile']
    dict_data["Median of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=50)['percentile']
    dict_data["Average tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3)["mean"]
    dict_data["75% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=75)['percentile']
    dict_data["90% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=90)['percentile']
    dict_data["95% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=95)['percentile']
    dict_data["99% of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3, p=99)['percentile']
    dict_data["Max of tokens"]=statistics_compute(hf_df1, hf_df2, hf_df3)["max"]
    token_count_df=pd.DataFrame(dict_data)
    return token_count_df

def style_format(token_count_df,  textbody="preprocessed_email"):
    token_count_df=token_count_df.set_index("data_type")
    token_count_df[list(token_count_df.columns)] = token_count_df[list(token_count_df.columns)].astype(int)
    return token_count_df.style.format("{:,}").set_caption(f"Summary Statistics of token lengths for {textbody} ").set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'red'),
            ('font-size', '20px')
        ]
    }])


# In[19]:


data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/","email-complaints","datasets")
train_df1=load_from_disk(os.path.join(data_dir,"train_df"))
val_df1=load_from_disk(os.path.join(data_dir,"val_df"))
test_df1=load_from_disk(os.path.join(data_dir,"test_df"))


# In[4]:


token_count_df=statistics_table(train_df1,val_df1,test_df1)
style_format(token_count_df,  textbody="preprocessed_email")


# In[7]:


train_df1.set_format("pandas")
df1_train=train_df1[:]

val_df1.set_format("pandas")
df1_val=val_df1[:]

test_df1.set_format("pandas")
df1_test=test_df1[:]

fig,(ax1,ax3)=plt.subplots(1,2, figsize=(25,6))
sns.histplot(df1_train['text_length'],ax=ax1)
ax1.set_title("email token length\n(Training Set)")
ax1.set_xlabel("token counts")
ax1.set_ylabel("")
ax1.set(xlim=(0, 20000))

# sns.histplot(df1_val['text_length'],ax=ax2)
# ax2.set_title("email token length\n(Validation Set)")
# ax2.set_xlabel("token counts")
# ax2.set_ylabel("")
# ax2.set(xlim=(0, 20000))
# plt.show()

sns.histplot(df1_test['text_length'],ax=ax3)
ax3.set_title("email token length\n(Test Set)")
ax3.set_xlabel("token counts")
ax3.set_ylabel("")
ax3.set(xlim=(0, 20000))
plt.show()


# In[8]:


fig,(ax1,ax2)=plt.subplots(1,2, figsize=(25,6))

df1_train.boxplot("text_length", by="is_complaint", grid=True, showfliers=False,color="black",ax=ax1)
ax1.set_title("token length per email \n(Training set)")
ax1.set_ylabel("# of token")

df1_test.boxplot("text_length", by="is_complaint", grid=True, showfliers=False,color="black",ax=ax2)
ax2.set_title("token length per email \n(Test set)")
ax2.set_ylabel("# of token")
plt.show()


# In[13]:


test_df=load_from_disk("./test_df")
test_df.set_format("pandas")
test_df=test_df[:]


# In[15]:


import textwrap
import random

df_test_v1=test_df[(test_df.is_complaint=="Y") & (test_df.text_length<=2000)]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = df_test_v1["preprocessed_email"]

# Randomly choose some examples.
for i in range(10):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
    print('')
    print(wrapper.fill(exam_text[j]))
    print('')


# In[ ]:





# In[ ]:





# In[41]:


import textwrap
import random

df_test_v1=test_df[(test_df.is_complaint=="Y") & (test_df.text_length<=200)]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = df_test_v1["preprocessed_email"]

# Randomly choose some examples.
for i in range(10):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
    print('')
    print(wrapper.fill(exam_text[j]))
    print('')


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


text="""
Now Redi-Bag is blank, see below. Nothing personal but I am getting really frustrated at Chase's inability to fix this. Subject: RE: Transfer
Complete- Redi Bag Inc Good Morning Sebastien, I am sorry that it didn't work. I reached out to support and we were able to turn it on from our end
this morning. Can you try again to confirm? Again, my apologies for the back and forth, but hopefully this fixes the problem. Let me know. Subject:
RE: Transfer Complete- Redi Bag Inc Hello Derrick, Unfortunately this is still not working. See below print screen. Subject: RE: Transfer Complete-
Redi Bag Inc Hey Sebastien, No problem, it was a pleasure working with you. I hope you had a great vacation. Were you able to see if positive pay is
working now? Subject: RE: Transfer Complete- Redi Bag Inc Thank you for all your help throughout that process Derrick. Sebastien Subject: Transfer
Complete- Redi Bag Inc Hello Sebastien , We welcome you to the Commercial Bank. Thank you for your time throughout this transfer process. We have
completed all required tasks for your transfer into Commercial Bank for Redi Bag Inc and related approved entities. Your dedicated relationship team
is listed below. For any service questions, please reach out to your Client Service contact. If you have previously used any Chase Branches to
initiate any of your transactions, please discontinue doing this and use your Connect Services going forward. Mark Long Banker  Olivia Mcleod-Smith
Treasury Management Officer  Chase Illinois Service Team/Tamara Gray Client Service Professional  Chase Connect Support Team Ph: 1-877-226-0071 For
Chase Connect you can access the Go to guides page - modern resource guides and short videos to acclimate you to the main features of Chase Connect.
For information on Chase Connect Fraud Protection Services (Positive Pay, Reverse Positive Pay, ACH Debit Block), please visit the following link here
It's been a pleasure working with you throughout this process. As a firm, we use feedback to drive continuous improvement. You may receive a brief
survey about your implementation experience. We'd appreciate your thoughts on what went well and how we can further improve the process. Should you
have any questions, please do not hesitate to reach out. We aim to exceed your expectations. Tell us how we are doing @ better together.
"""


# In[2]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
model_checkpoint=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","longformer-large-4096")

tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(model_checkpoint)


# In[4]:


import torch
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits


# In[5]:


logits


# In[6]:


predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]


# In[12]:


inputs = tokenizer("It is awful to see this great movie is over", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
logits


# In[11]:


inputs = tokenizer("It is great to see this awful movie is over", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
logits


# In[ ]:




