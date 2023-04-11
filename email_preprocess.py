#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/smarsh-emails/combined/preprocessed/model_training/splits  ./datasets \
# --recursive --exclude "*" --include "*.csv" --exclude "*/*"


# In[3]:


# !aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/R680994/notebooks/complaints_larger_instance/data/  ./misc/data/  \
# --recursive --exclude "*" --include "*.csv" --include "*.csv.zip" --exclude "*/*"


# In[1]:


get_ipython().system('aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/smarsh-emails/combined/preprocessed/model_training/splits/train.csv  ./datasets/')
get_ipython().system('aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/smarsh-emails/combined/preprocessed/model_training/splits/val.csv  ./datasets/')
get_ipython().system('aws s3 cp s3://app-id-105879-dep-id-105006-uu-id-owjkztywmzkt/smarsh-emails/combined/preprocessed/model_training/splits/test.csv  ./datasets/')


# In[ ]:





# In[34]:


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


train_df["front office"].value_counts(dropna=False)


# In[5]:


# plt.rcParams["figure.figsize"] = [10, 5]
# plt.rcParams["figure.autolayout"] = True
# ax = sns.barplot(data = train_df, x='service',y='is_complaint')
# ax.set_title(" termination word existing in email")
# plt.legend(loc="best")


# In[ ]:





# In[28]:


hf_train=Dataset.from_pandas(train_df)
hf_val=Dataset.from_pandas(val_df)
hf_test=Dataset.from_pandas(test_df)

hf_data=DatasetDict({"train":hf_train, "val":hf_val,  "test":hf_test})


# In[29]:


hf_data


# In[30]:


def label_distribution(df):
    tempt1=pd.DataFrame(df["is_complaint"].value_counts(dropna=False)).reset_index().rename(columns={'index':'is_complaint','is_complaint':'count'})
    tempt2=pd.DataFrame(df["is_complaint"].value_counts(dropna=False,normalize=True)).reset_index().rename(columns={'index':'is_complaint','is_complaint':'percentage'})
    return tempt1.merge(tempt2, on="is_complaint", how="inner")

def style_format(df,  data_type="Training set"):
    return df.style.format({'count':'{:,}','percentage':'{:.2%}'})           .set_caption(f"{data_type} label distribution")           .set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}])


# In[31]:


label_train=label_distribution(train_df)
style_format(label_train,  data_type="Training set")


# In[32]:


label_val=label_distribution(val_df)
style_format(label_val,  data_type="validation set")


# In[33]:


label_test=label_distribution(test_df)
style_format(label_test,  data_type="Test set")


# In[ ]:


0    precision : 100%,    Recall: 76%
1    precision : 0.98% ,  recall: 85%
2                0.3%     recall: 


# In[13]:


model_checkpoint=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","roberta-base")
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)


# In[15]:


train_df['preprocessed_email'] = train_df['preprocessed_email'].astype(str)
val_df['preprocessed_email'] = val_df['preprocessed_email'].astype(str)
test_df['preprocessed_email'] = test_df['preprocessed_email'].astype(str)


# In[16]:


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

data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/","email-complaints","datasets")
train_df1.save_to_disk(os.path.join(data_dir,"train_df"))
val_df1.save_to_disk(os.path.join(data_dir,'val_df'))
test_df1.save_to_disk(os.path.join(data_dir,'test_df'))


# In[20]:


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


# In[35]:


data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/","email-complaints","datasets")
train_df1=load_from_disk(os.path.join(data_dir,"train_df"))
val_df1=load_from_disk(os.path.join(data_dir,"val_df"))
test_df1=load_from_disk(os.path.join(data_dir,"test_df"))


# In[36]:


def convert_hf_pandas(df):
    df.set_format(type="pandas")
    data=df[:]
    return data

train_data=convert_hf_pandas(train_df1)
val_data=convert_hf_pandas(val_df1)
test_data=convert_hf_pandas(test_df1)

train_data["data_type"]=["training_set"]*len(train_data)
val_data["data_type"]=["validation_set"]*len(val_data)
test_data["data_type"]=["test_set"]*len(test_data)
all_data=pd.concat([train_data,val_data,test_data],axis=0)


# In[37]:


all_data["is_complaint"]=all_data["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)

plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
ax = sns.barplot(data = all_data, x='service',y='is_complaint',hue="data_type")
ax.set_title("sevice vs non-service email")
plt.legend(loc="best")


# In[13]:


plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
ax = sns.barplot(data = all_data, x='front office',y='is_complaint',hue="data_type")
ax.set_title("front office vs non-front office email")
plt.legend(loc="best")


# In[38]:


def pcut_func(df,var,nbin=5):
    df[var]=df[var].astype(float)
    df["cut"]=pd.qcut(df[var],nbin,precision=2,duplicates="drop")
    decile=df.groupby(df["cut"])['target'].mean().reset_index()
    decile["cut"]=decile["cut"].astype(str)
    return decile

# def myplot(df,var,*args):

#     fig, a = plt.subplots(len(args)//2,2,figsize=(12,2.5*len(args)))
#     a=a.ravel()
#     for idx,ax in enumerate(a):
#         df=args[idx]
#         ax.plot(df["cut"],df["churn"],color="r",marker="*",linewidth=2, markersize=12)
#         ax.set_title(var[idx])
#         ax.tick_params(labelrotation=45)
#     fig.tight_layout()
    
# variable_list=["text_length","issue_counts","duration","negative_word_counts"]
# nbin=5
# args=[]
# for idx,v in enumerate(variable_list):
#     x=pcut_func(train_df,var=variable_list[idx],nbin=nbin)
#     args.append(x)
# myplot(train_df,variable_list,*args)


# In[39]:


fig, ax = plt.subplots(1,3,figsize=(12,4))
plt.subplot(1,3,1)
df=pcut_func(train_data,var="text_length",nbin=10)
ax[0].plot(df["cut"],df["target"],color="r",marker="*",linewidth=2, markersize=12)
ax[0].set_title("text_length\n(training set)")
ax[0].tick_params(labelrotation=45)
plt.subplot(1,3,2)
df=pcut_func(val_data,var="text_length",nbin=10)
ax[1].plot(df["cut"],df["target"],color="r",marker="*",linewidth=2, markersize=12)
ax[1].set_title("text_length\n(validation set)")
ax[1].tick_params(labelrotation=45)
plt.subplot(1,3,3)
df=pcut_func(test_data,var="text_length",nbin=10)
ax[2].plot(df["cut"],df["target"],color="r",marker="*",linewidth=2, markersize=12)
ax[2].set_title("text_length\n(test set)")
ax[2].tick_params(labelrotation=45)

fig.tight_layout()


# In[ ]:


fig, a = plt.subplots(1,3,figsize=(12,2.5*len(args)))
a=a.ravel()
for idx,ax in enumerate(a):
    df=args[idx]
    ax.plot(df["cut"],df["churn"],color="r",marker="*",linewidth=2, markersize=12)
    ax.set_title(var[idx])
    ax.tick_params(labelrotation=45)
fig.tight_layout()


# In[17]:


train_data.head(2)


# In[ ]:





# In[22]:


token_count_df=statistics_table(train_df1,val_df1,test_df1)
style_format(token_count_df,  textbody="preprocessed_email")


# In[23]:


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


# In[24]:


fig,(ax1,ax2)=plt.subplots(1,2, figsize=(25,6))

df1_train.boxplot("text_length", by="is_complaint", grid=True, showfliers=False,color="black",ax=ax1)
ax1.set_title("token length per email \n(Training set)")
ax1.set_ylabel("# of token")

df1_test.boxplot("text_length", by="is_complaint", grid=True, showfliers=False,color="black",ax=ax2)
ax2.set_title("token length per email \n(Test set)")
ax2.set_ylabel("# of token")
plt.show()


# In[28]:


data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/","email-complaints","datasets")

test_df=load_from_disk((os.path.join(data_dir,"test_df")))
test_df.set_format("pandas")
test_df=test_df[:]


# In[31]:


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


# In[32]:


df_test_v1=test_df[(test_df.is_complaint=="Y") & (test_df.text_length<=200)]
df_test_v1.head(2)


# In[33]:


import textwrap
import random

df_test_v1=test_df[(test_df.is_complaint=="Y") & (test_df.text_length<=200)]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = df_test_v1.loc[:,["snapshot_id","preprocessed_email"]]

# Randomly choose some examples.
for i in range(10):
    random.seed(101+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
    print('')
    print(exam_text.loc[j,"snapshot_id"])
    print()
    print(wrapper.fill(exam_text.loc[j,"preprocessed_email"]))
    print('')


# In[ ]:





# In[39]:


import textwrap
import random

df_test_v1=test_df[(test_df.is_complaint=="Y") & (test_df.text_length<=300)]

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = df_test_v1.loc[:,["snapshot_id","preprocessed_email"]]

# Randomly choose some examples.
for i in range(10):
    random.seed(103+i)
    j = random.choice(exam_text.index)
    
    print('')
    print("*"*50)
    print('*********  preprocessed_email ********')
    print("*"*50)
    print('')
    print(exam_text.loc[j,"snapshot_id"])
    print()
    print(wrapper.fill(exam_text.loc[j,"preprocessed_email"]))
    print('')


# In[ ]:




