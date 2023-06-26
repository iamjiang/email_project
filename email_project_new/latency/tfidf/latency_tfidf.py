import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF/')
sys.path=list(set(sys.path))

import csv
csv.field_size_limit(sys.maxsize)
import time
import os
import re
import pandas as pd
import numpy as np
import argparse
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
# Filter out the stopwords from the sentence
# filtered_words = [word for word in words if word.lower() not in stopwords_list]

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

from collections import Counter

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

import joblib

import utils

import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning, module='sklearn')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TFIDF+Classifier')

    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    
    parser.add_argument('--model_name', type=str, default="xgboost")
    parser.add_argument("--test_date", type=str, default="04_23", help="the month for test set")
    parser.add_argument("--device", default="cpu", type=str)
    args= parser.parse_args()

    print()
    print(args)
    print()
    
    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF", "tfidf_data", args.test_date)
    df_train=pd.read_pickle(os.path.join(data_dir,"train_data_pickle"))
    
    y_train=df_train.loc[:,["target_variable"]]
    index_train=df_train.loc[:,["snapshot_id","thread_id","time_variable"]]
    x_train=df_train.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)
    
    data_path="/opt/omniai/work/instance1/jupyter/v4_new_email/latency/tfidf/tfidf_data/"
        
    df_test=pd.read_pickle(os.path.join(data_path,"test_data_pickle"))

    print()
    print(df_test["target_variable"].value_counts(dropna=False))
    print()
    
    y_test=df_test.loc[:,["target_variable"]]
    index_test=df_test.loc[:,["snapshot_id","thread_id","time_variable"]]
    x_test=df_test.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)
    
    cat_features_names = ["negative_word"]
    
    encoder = LabelEncoder()
    
    output_dir=os.getcwd()
            
    # if args.model_name=="catboost":
    #     model_dir=os.path.join(os.getcwd(),"model")
    #     cat_model = joblib.load(os.path.join(model_dir,'cat_model.pkl'))
        
    if args.model_name=="lightgbm":
        model_dir=os.path.join(os.getcwd(),"model")
        lgb_model = joblib.load(os.path.join(model_dir,'lgb_model.pkl'))
        for feature in cat_features_names:
            x_train[feature] = encoder.fit_transform(x_train[feature])
            x_test[feature] = encoder.transform(x_test[feature])
        
        start_time=time.time()
        test_pred=lgb_model.predict(x_test)
        end_time=time.time()
        duration=end_time-start_time
        total_inputs=x_test.shape[0]
        throughput=total_inputs/duration
        latency=duration/total_inputs # multiply 1000 to convert second into milliseconds per input   
        
        with open(os.path.join(output_dir,"latency_throughput.txt"),'a') as f:
            f.write(f'{args.model_name},{latency},{throughput},{duration},{args.device}\n')
        print()  
        print("model_name: {:} | latency: {:.4f} | throughput: {:.4f} | duration: {:.4f} | device: {:} ".\
              format(args.model_name, latency, throughput, duration, args.device))    
        print()
        
    elif args.model_name=="xgboost":
        model_dir=os.path.join(os.getcwd(),"model")
        xgb_model = joblib.load(os.path.join(model_dir,'xgb_model.pkl'))
        for feature in cat_features_names:
            x_train[feature] = encoder.fit_transform(x_train[feature])
            x_test[feature] = encoder.transform(x_test[feature])
        
        start_time=time.time()
        test_pred=xgb_model.predict(xgb.DMatrix(x_test))
        end_time=time.time()
        duration=end_time-start_time
        total_inputs=x_test.shape[0]
        throughput=total_inputs/duration
        latency=duration/total_inputs # multiply 1000 to convert second into milliseconds per input
        
        with open(os.path.join(output_dir,"latency_throughput.txt"),'a') as f:
            f.write(f'{args.model_name},{latency},{throughput},{duration},{args.device}\n')
        print()  
        print("model_name: {:} | latency: {:.4f} | throughput: {:.4f} | duration: {:.4f} | device: {:} ".\
              format(args.model_name, latency, throughput,duration, args.device))    
        print()        
        
    elif args.model_name=="randomforest":
        model_dir=os.path.join(os.getcwd(),"model")
        rf_model = joblib.load(os.path.join(model_dir,'rf_model.pkl'))
        for feature in cat_features_names:
            x_train[feature] = encoder.fit_transform(x_train[feature])
            x_test[feature] = encoder.transform(x_test[feature])
        
        start_time=time.time()
        test_pred=rf_model.predict_proba(x_test)[:,1]  
        end_time=time.time()
        duration=end_time-start_time
        total_inputs=x_test.shape[0]
        throughput=total_inputs/duration
        latency=duration/total_inputs # multiply 1000 to convert second into milliseconds per input 
        
        with open(os.path.join(output_dir,"latency_throughput.txt"),'a') as f:
            f.write(f'{args.model_name},{latency},{throughput},{duration},{args.device}\n')
        print()  
        print("model_name: {:} | latency: {:.4f} | throughput: {:.4f} | duration: {:.4f} | device: {:} "\
              .format(args.model_name, latency, throughput, duration, args.device))    
        print()        
        
    else:
        raise ValueError("Invalid model name. Only catboost, lightgbm , xgboost or randomforest are support")
