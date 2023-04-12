import time
import os
import re
import operator
import pickle
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
import itertools
import spacy
model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","en_core_web_md","en_core_web_md-3.3.0")
nlp = spacy.load(model_name)
# from textblob import TextBlob
# python -m textblob.download_corpora
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
nltk.data.path.append(nltk_data_dir)
# # Filter out the stopwords from the sentence
# filtered_words = [word for word in words if word.lower() not in stopwords_list]

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

from collections import Counter

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from catboost import CatBoostClassifier, Pool
import utils

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def text_preprocess(text, extract_adj=False):
    # lemma = nltk.wordnet.WordNetLemmatizer()
    text = str(text)
    #remove http links from the email
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], '')  
    text = re.sub("`", "'", text)
    if extract_adj:
        ADJ_word=[]
        doc=nlp(text)
        for token in doc:
            if token.pos_=="ADJ":
                ADJ_word.append(token.text)   
        text=" ".join(ADJ_word)    
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

if __name__=="__main__":
    # df_train=pd.read_pickle(os.path.join(os.getcwd(),"df_train"))
    # df_val=pd.read_pickle(os.path.join(os.getcwd(),"df_val"))
    # df_test=pd.read_pickle(os.path.join(os.getcwd(),"df_test"))
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","datasets")
    train_val_test=pd.read_pickle(os.path.join(data_path,"train_val_test_pickle"))
    train_val_test.loc[:,'target']=train_val_test.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0) 
    train_val_test.loc[:,"text_length"]=train_val_test.loc[:,"preprocessed_email"].progress_apply(lambda x: len(x.split()))
    
    train_val_test["bag_of_word"]=train_val_test["preprocessed_email"].progress_apply(text_preprocess)
    words = set(nltk.corpus.words.words())

    train_val_test["bag_of_word"] = train_val_test["bag_of_word"]\
    .progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))   
    
    df_train=train_val_test[train_val_test["data_type"]=="training_set"].drop(["data_type"],axis=1)
    df_val=train_val_test[train_val_test["data_type"]=="validation_set"].drop(["data_type"],axis=1)
    df_test=train_val_test[train_val_test["data_type"]=="test_set"].drop(["data_type"],axis=1)
    
    negative_word=[]
    with open("negative-words.txt") as f:
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

    word=set(word_train_complaint[0:20]).difference(set(word_train_no_ccomplaint[0:50]))
    print()
    print(word)
    print()
    # tempt["negative_word"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(word).intersection(set(x.split())))!=0 else 0 )
    df_train["negative_word"]=df_train["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
    df_val["negative_word"]=df_val["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
    df_test["negative_word"]=df_test["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )

    def refine_data(df):
        x=df.loc[:,["service","front office","negative_word","text_length"]]
        x["service"].replace({True:1,False:0},inplace=True)
        x["front office"].replace({True:1,False:0},inplace=True)
        y=df.loc[:,["target"]]
        x["service"]=x["service"].astype(str)
        x["front office"]=x["front office"].astype(str)
        x["negative_word"]=x["negative_word"].astype(str)
        # x["text_length_decile"]=x["text_length_decile"].astype(str)
        x.rename(columns={"service": "service_feature", "front office": "front_office"},inplace=True) ## rename these words to distinguish them from the tfidf fetures
        y.rename(columns={"target": "target_variable"},inplace=True) ## rename these words to distinguish them from the tfidf fetures
        return x,y
    
    x_train,y_train=refine_data(df_train)
    x_val,y_val=refine_data(df_val)
    x_test,y_test=refine_data(df_test)
    
    ################## TFIDF ######################
    
    max_feature_num=7000
    
    bow_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=2, max_features=max_feature_num, stop_words='english')
    # bag-of-words feature matrix
    # bow = bow_vectorizer.fit_transform(df_train['preprocessed_email'])
    bow = bow_vectorizer.fit_transform(df_train["bag_of_word"])
    train_tfidf = bow.toarray()
    vocab = bow_vectorizer.vocabulary_.keys()
    vocab = list(vocab)
    train_tfidf = pd.DataFrame(train_tfidf,columns=vocab)

    # val_tfidf = bow_vectorizer.transform(df_val['preprocessed_email'])
    val_tfidf = bow_vectorizer.transform(df_val["bag_of_word"])
    val_tfidf = val_tfidf.toarray()
    val_tfidf = pd.DataFrame(val_tfidf,columns=vocab)

    # test_tfidf = bow_vectorizer.transform(df_test['preprocessed_email'])
    test_tfidf = bow_vectorizer.transform(df_test["bag_of_word"])
    test_tfidf = test_tfidf.toarray()
    test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)

    train_data=pd.concat([train_tfidf.reset_index(drop=True),x_train.reset_index(drop=True), y_train.reset_index(drop=True)],axis=1)
    val_data=pd.concat([val_tfidf.reset_index(drop=True),x_val.reset_index(drop=True), y_val.reset_index(drop=True)],axis=1)
    test_data=pd.concat([test_tfidf.reset_index(drop=True),x_test.reset_index(drop=True), y_test.reset_index(drop=True)],axis=1)
    
    
    data_dir=os.path.join(os.getcwd(),"tfidf+structure","max_feat_7000")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    train_data.to_pickle(os.path.join(data_dir,"train_data_pickle"))
    val_data.to_pickle(os.path.join(data_dir,"val_data_pickle"))
    test_data.to_pickle(os.path.join(data_dir,"test_data_pickle"))
    
#     data_dir=os.path.join(os.getcwd(),"tfidf+structure")
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
        
#     train_data.to_pickle(os.path.join(data_dir,"train_data_pickle"))
#     val_data.to_pickle(os.path.join(data_dir,"val_data_pickle"))
#     test_data.to_pickle(os.path.join(data_dir,"test_data_pickle"))
                                              

#     train_data=pd.read_pickle(os.path.join(data_dir,"train_data_pickle"))
#     val_data=pd.read_pickle(os.path.join(data_dir,"val_data_pickle"))
#     test_data=pd.read_pickle(os.path.join(data_dir,"test_data_pickle"))
    
