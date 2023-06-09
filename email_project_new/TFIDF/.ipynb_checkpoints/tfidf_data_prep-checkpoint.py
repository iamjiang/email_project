import argparse
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment=None
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML
import re
import textwrap
import random

import string
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Load the stopwords from the new directory
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
STOPWORDS=[x.strip() for x in stopwords_list]
nltk.data.path.append(nltk_data_dir)

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from catboost import CatBoostClassifier, Pool

def val_mask_creation(dataset,target_variable, validation_split):
    
    dataset.sort_values(by='time', ascending=False, axis=0, inplace = True)
    dataset=dataset.reset_index(drop=True)
    
    train_idx=[]
    val_idx=[]
    
    LABEL=dataset[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    target_list=np.unique(LABEL).tolist()
        
    for i in range(len(target_list)):
        
        _idx=IDX[LABEL==target_list[i]]
        ## split train and valiation by time instead of randomly
        # np.random.seed(seed)
        # np.random.shuffle(_idx)
        
        split=int(np.floor(validation_split*_idx.shape[0]))
        
        val_idx.extend(_idx[ : split])
        print(len(_idx[ : split]))
        train_idx.extend(_idx[split:])        
 
    all_idx=np.arange(LABEL.shape[0])

    val_idx=np.array(val_idx)
    train_idx=np.array(train_idx)
    
    df_train=dataset.loc[train_idx,:]
    df_train=df_train.reset_index(drop=True)
    
    df_val=dataset.loc[val_idx,:]
    df_val["data_type"]=["val"]*val_idx.shape[0]
    df_val=df_val.reset_index(drop=True)
    
    return df_train, df_val

def bow_preprocess(text):
    # lemma = nltk.wordnet.WordNetLemmatizer()
    text = str(text) 
    ### Remove stop word
    text = [word for word in word_tokenize(text) if word.lower() not in STOPWORDS]
    text = " ".join(text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    #Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split()]
    text=" ".join(text)
    return text  

def main(df,args):
    
    df["bag_of_word"]=df["preprocessed_email"].progress_apply(bow_preprocess)
    words = set(nltk.corpus.words.words())
    df["bag_of_word"] = df["bag_of_word"]\
    .progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))

    df_train=df[df["data_type"]=="train"]
    df_val=df[df["data_type"]=="val"]
    df_test=df[df["data_type"]=="test"]    
    
    negative_word=[]
    with open(os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF","negative-words.txt")) as f:
        for curline in f:
            if curline.startswith(";"):
                continue
            if curline.strip():
                negative_word.append(curline.strip())

    print()
    print("There are {:,} negative words externally".format(len(negative_word)))
    print()
    
    df_train['negative_word_set']=df_train["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))
    df_val['negative_word_set']=df_val["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))
    df_test['negative_word_set']=df_test["bag_of_word"].progress_apply(lambda x: set(x.split()).intersection(set(negative_word)))

    train_complaint,  train_no_complaint=df_train[df_train['target']==1], df_train[df_train['target']==0]
    val_complaint,  val_no_complaint=df_val[df_val['target']==1], df_val[df_val['target']==0]
    test_complaint,  test_no_complaint=df_test[df_test['target']==1], df_test[df_test['target']==0]

    def most_common_word(df,feature):
        word_count=Counter()
        for index,row in tqdm(df.iterrows(), total=df.shape[0]):
            if isinstance(row[feature],list):
                word_count.update(set(row[feature].split()))
            elif isinstance(row[feature],set):
                word_count.update(row[feature])
        word,freq=zip(*word_count.most_common())
        return word,freq

    word_train_complaint, freq_train_complaint = most_common_word(train_complaint, feature="negative_word_set")
    word_val_complaint, freq_val_complaint = most_common_word(val_complaint, feature="negative_word_set")
    word_test_complaint, freq_test_complaint = most_common_word(test_complaint, feature="negative_word_set")

    word_train_no_ccomplaint, freq_train_no_complaint = most_common_word(train_no_complaint, feature="negative_word_set")
    word_val_no_ccomplaint, freq_val_no_complaint = most_common_word(val_no_complaint, feature="negative_word_set")
    word_test_no_ccomplaint, freq_test_no_complaint = most_common_word(test_no_complaint, feature="negative_word_set")

    word=set(word_train_complaint[0:50]).difference(set(word_train_no_ccomplaint[0:50]))
    print()
    print(word)
    print()
    
    df_train["negative_word"]=df_train["bag_of_word"].\
    progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
    df_val["negative_word"]=df_val["bag_of_word"].\
    progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
    df_test["negative_word"]=df_test["bag_of_word"].\
    progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
    
    def refine_data(df):
        x=df.loc[:,["negative_word","text_length"]]
        y=df.loc[:,["target"]]
        x["negative_word"]=x["negative_word"].astype(str)
        # x["text_length_decile"]=x["text_length_decile"].astype(str)
        y.rename(columns={"target": "target_variable"},inplace=True) ## rename these words to distinguish them from the tfidf possible fetures: target
        df.rename(columns={"time": "time_variable"},inplace=True) ## rename these words to distinguish them from the tfidf possible fetures: time
        index_data=df.loc[:,['snapshot_id','thread_id','time_variable']]
        return x,y,index_data
    
    x_train,y_train,index_train=refine_data(df_train)
    x_val,y_val,index_val=refine_data(df_val)
    x_test,y_test,index_test=refine_data(df_test)    
    
    ################## TFIDF ######################
    
    bow_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=2, max_features=args.max_feature_num, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(df_train['preprocessed_email'])
    # bow = bow_vectorizer.fit_transform(df_train["bag_of_word"])
    train_tfidf = bow.toarray()
    vocab = bow_vectorizer.vocabulary_.keys()
    vocab = list(vocab)
    train_tfidf = pd.DataFrame(train_tfidf,columns=vocab)

    val_tfidf = bow_vectorizer.transform(df_val['preprocessed_email'])
    # val_tfidf = bow_vectorizer.transform(df_val["bag_of_word"])
    val_tfidf = val_tfidf.toarray()
    val_tfidf = pd.DataFrame(val_tfidf,columns=vocab)

    test_tfidf = bow_vectorizer.transform(df_test['preprocessed_email'])
    # test_tfidf = bow_vectorizer.transform(df_test["bag_of_word"])
    test_tfidf = test_tfidf.toarray()
    test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)
    
    train_data=pd.concat([train_tfidf.reset_index(drop=True),x_train.reset_index(drop=True), y_train.reset_index(drop=True), index_train.reset_index(drop=True)],axis=1)
    val_data=pd.concat([val_tfidf.reset_index(drop=True),x_val.reset_index(drop=True), y_val.reset_index(drop=True), index_val.reset_index(drop=True)],axis=1)
    test_data=pd.concat([test_tfidf.reset_index(drop=True),x_test.reset_index(drop=True), y_test.reset_index(drop=True), index_test.reset_index(drop=True)],axis=1)    
    
    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF", args.output_dir, args.test_date)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    train_data.to_pickle(os.path.join(data_dir,"train_data_pickle"))
    val_data.to_pickle(os.path.join(data_dir,"val_data_pickle"))
    test_data.to_pickle(os.path.join(data_dir,"test_data_pickle"))
    
if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_feature_num', type=int, default=7000)
    argparser.add_argument('--output_dir', type=str, default=None)
    argparser.add_argument('--validation_split', type=float, default=0.2)
    argparser.add_argument("--closed_status", action="store_true", help="only keep status=closed") 
    argparser.add_argument("--test_date", type=str, default="04_23", help="the month for test set")
    
    args,_ = argparser.parse_known_args()

    args.output_dir="tfidf_data"
        
    print(args)
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v4_new_email","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))
    
    if args.closed_status:
        ### only keep emails with status=closed
        df=df[df.state=="closed"]
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    if args.test_date=="04_23":
        ## train: 09/2022 ~ 02/2023. validation: 03/2023  test: 04/2023
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2,3]) else "test"
    elif args.test_date=="03_23":
        df=df[df['time']<'2023-04-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2]) else "test"
    elif args.test_date=="02_23":
        df=df[df['time']<'2023-03-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1]) else "test"        
    else:
        raise ValueError("Invalid test_date. Only 02_23,03_23,04_23 are support currently")
        
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    # df.loc[:,'target']=df.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']=="Y"),1,0)
    
    df1=df[df.data_type=="train"]
    df1=df1.reset_index(drop=True)
    df_train,df_val=val_mask_creation(df1,'target', validation_split=args.validation_split)
    
    df_test=df[df.data_type=="test"]
    df_test=df_test.reset_index(drop=True)
    ## overwrite the target with the ground true complaint label
    df_test['target']=df_test['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    df=pd.concat([df_train,df_val,df_test],axis=0,ignore_index=True)
    df=df.reset_index(drop=True)
    
    main(df,args)


    