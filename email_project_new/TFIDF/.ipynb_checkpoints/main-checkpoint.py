import sys
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
# model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","en_core_web_md","en_core_web_md-3.3.0")
# nlp = spacy.load(model_name)
# from textblob import TextBlob
# python -m textblob.download_corpora
import string
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Load the stopwords from the new directory
# nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
# stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
# stopwords_list = stopwords_file.readlines()
# nltk.data.path.append(nltk_data_dir)
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

def catboost_traning(x_train,y_train,x_val,y_val,x_test,y_test,\
                     cat_features_names = ["negative_word"],target="target_variable"):
    
    
    cat_features = [x_train.columns.get_loc(col) for col in cat_features_names]
    
    train_data = Pool(data=x_train,
                     label=y_train,
                     cat_features=cat_features
                     )
    val_data = Pool(data=x_val,
                    label=y_val,
                    cat_features=cat_features
                   )
    
    train_label=y_train[target].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    train_classes_num, loss_weight = utils.get_class_count_and_weight(train_label,num_classes)
    
    params = {'loss_function':'Logloss',
          'eval_metric':"PRAUC",
          'iterations': 1000,
          'learning_rate': 2e-5,
#           'cat_features': cat_features, # we don't need to specify this parameter as 
#                                           pool object contains info about categorical features
          'early_stopping_rounds': 50,
          'verbose': 200,
          'random_seed': 101,
          'scale_pos_weight': float(loss_weight[1]/loss_weight[0])
         }

    clf_model = CatBoostClassifier(**params)
    clf_model.fit(train_data, # instead of X_train, y_train
              eval_set=val_data, # instead of (X_valid, y_valid)
              use_best_model=True, 
              plot=True
             );
    
    train_pred=clf_model.predict_proba(x_train)[:,1]
    val_pred=clf_model.predict_proba(x_val)[:,1]
    test_pred=clf_model.predict_proba(x_test)[:,1]
    
    return clf_model, train_pred, val_pred, test_pred

    # model_dir=os.path.join(os.getcwd(),"tfidf+structure")
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # joblib.dump(clf_model, os.path.join(model_dir,'clf.pkl'))
    # clf_model = joblib.load(os.path.join(model_dir,'clf.pkl'))
    

    
def lightgbm_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                      cat_features_names = ["negative_word"],target="target_variable"):
    
    train_label=y_train[target].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    train_classes_num, loss_weight = utils.get_class_count_and_weight(train_label,num_classes)

    cat_features = [x_train.columns.get_loc(col) for col in cat_features_names]
    encoder = LabelEncoder()
    for feature in cat_features_names:
        x_train[feature] = encoder.fit_transform(x_train[feature])
        x_val[feature] = encoder.transform(x_val[feature])
        x_test[feature] = encoder.transform(x_test[feature])
        
    train_data=lgb.Dataset(x_train, label=y_train,categorical_feature=cat_features)
    val_data=lgb.Dataset(x_val, label=y_val,categorical_feature=cat_features)
    
    params = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "num_threads": 4,
    "scale_pos_weight": float(loss_weight[1]/loss_weight[0]),
    "seed":101
    }

    # Train LightGBM model on concatenated dataset
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        early_stopping_rounds=50,
    )
    
    train_pred=lgb_model.predict(x_train)
    val_pred=lgb_model.predict(x_val)
    test_pred=lgb_model.predict(x_test)
    
    return lgb_model, train_pred, val_pred, test_pred


def xgboost_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                     cat_features_names = ["negative_word"],target="target_variable"):
    
    train_label=y_train[target].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    train_classes_num, loss_weight = utils.get_class_count_and_weight(train_label,num_classes)

    cat_features = [x_train.columns.get_loc(col) for col in cat_features_names]
    encoder = LabelEncoder()
    for feature in cat_features_names:
        x_train[feature] = encoder.fit_transform(x_train[feature])
        x_val[feature] = encoder.transform(x_val[feature])
        x_test[feature] = encoder.transform(x_test[feature])
        
    # Convert data into DMatrix format
    train_data = xgb.DMatrix(x_train, label=y_train)
    val_data = xgb.DMatrix(x_val, label=y_val)

    
    params = {
    "objective": "binary:logistic",
    "metric": "aucpr",
    "max_depth": 6,
    "eta": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": float(loss_weight[1]/loss_weight[0]),
    "seed": 101
    }

    xgb_model = xgb.train(
        params,
        train_data,
        num_boost_round=1000,
        evals=[(val_data, "Validation")],
        early_stopping_rounds=50,
        verbose_eval=10
    )
    
    train_pred=xgb_model.predict(xgb.DMatrix(x_train))
    val_pred=xgb_model.predict(xgb.DMatrix(x_val))
    test_pred=xgb_model.predict(xgb.DMatrix(x_test))
    
    return xgb_model, train_pred, val_pred, test_pred


def randomforest_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                          cat_features_names = ["negative_word"]):

    cat_features = [x_train.columns.get_loc(col) for col in cat_features_names]
    encoder = LabelEncoder()
    for feature in cat_features_names:
        x_train[feature] = encoder.fit_transform(x_train[feature])
        x_val[feature] = encoder.transform(x_val[feature])
        x_test[feature] = encoder.transform(x_test[feature])
        
    # Define hyperparameters to vary for validation curve
    param_grid = {"n_estimators": [50,100, 200],"max_depth": [3, 6, 9]}

    # Tune hyperparameters using GridSearchCV
    rf_model = BalancedRandomForestClassifier(
        class_weight='balanced_subsample', random_state=101
    )
    grid_search = GridSearchCV(
        rf_model, param_grid=param_grid, cv=5, scoring="average_precision", n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    
    rf_model= grid_search.best_estimator_
    
    train_pred=rf_model.predict_proba(x_train)[:,1]
    val_pred=rf_model.predict_proba(x_val)[:,1]
    test_pred=rf_model.predict_proba(x_test)[:,1]    
    
    return rf_model, train_pred, val_pred, test_pred

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TFIDF+Classifier')

    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")

    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.8)
    parser.add_argument('--bagging_freq', type=int, default=5)

    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help="early stop rounds for lightgbm")
    
    parser.add_argument('--model_name', type=str, default="xgboost")
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--test_date", type=str, default="04_23", help="the month for test set")
    
    parser.add_argument("--default_threshold", action="store_true", help="undersampling or not")
    
    args= parser.parse_args()
    
    # args.train_undersampling=True

    print()
    print(args)
    print()
    

    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF", "tfidf_data", args.test_date)

    df_train=pd.read_pickle(os.path.join(data_dir,"train_data_pickle"))
    df_val=pd.read_pickle(os.path.join(data_dir,"val_data_pickle"))
    df_test=pd.read_pickle(os.path.join(data_dir,"test_data_pickle"))
    
    if args.train_undersampling:
        df_train=utils.under_sampling(df_train,"target_variable",seed=args.seed, negative_positive_ratio=args.train_negative_positive_ratio)
    if args.val_undersampling:
        df_var=utils.under_sampling(df_val,"target_variable",seed=args.seed, negative_positive_ratio=args.val_negative_positive_ratio)

    y_train=df_train.loc[:,["target_variable"]]
    index_train=df_train.loc[:,["snapshot_id","thread_id","time_variable"]]
    x_train=df_train.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)

    y_val=df_val.loc[:,["target_variable"]]
    index_val=df_val.loc[:,["snapshot_id","thread_id","time_variable"]]
    x_val=df_val.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)

    y_test=df_test.loc[:,["target_variable"]]
    index_test=df_test.loc[:,["snapshot_id","thread_id","time_variable"]]
    x_test=df_test.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)
    
    cat_features_names = ["negative_word"]
    
    if args.model_name=="catboost":
        clf_model, train_pred, val_pred, test_pred=catboost_traning(x_train,y_train,x_val,y_val,x_test,y_test,\
                                                                    cat_features_names=cat_features_names,target="target_variable")
    elif args.model_name=="lightgbm":
        lgb_model, train_pred, val_pred, test_pred=lightgbm_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                                                                     cat_features_names=cat_features_names,target="target_variable")
    elif args.model_name=="xgboost":
        xgb_model, train_pred, val_pred, test_pred=xgboost_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                                                                    cat_features_names=cat_features_names,target="target_variable")
        
    elif args.model_name=="randomforest":
        rf_model, train_pred, val_pred, test_pred=randomforest_training(x_train,y_train,x_val,y_val,x_test,y_test,\
                                                                        cat_features_names=cat_features_names)
    else:
        raise ValueError("Invalid model name. Only catboost, lightgbm , xgboost or randomforest are support")
    
    # train_pred=model.predict_proba(x_train)[:,1]
    # val_pred=model.predict_proba(x_val)[:,1]
    # test_pred=model.predict_proba(x_test)[:,1]

    if args.default_threshold:
        best_threshold=0.5
    else:
        best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=args.val_min_recall, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    test_output=utils.model_evaluate(y_test.values.reshape(-1),test_pred.squeeze(),best_threshold)

    output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v4_new_email/TFIDF", "results", args.test_date, args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    text_length=x_test["text_length"].tolist()
    snapshot_id=index_test["snapshot_id"].tolist()
    thread_id=index_test["thread_id"].tolist()
    time=index_test["time_variable"].tolist()

    # fieldnames = ['True label', 'Predicted label', 'Predicted_prob']
    fieldnames = ['snapshot_id','thread_id','time','text_length','True_label', 'Predicted_label', 'Predicted_prob','best_threshold']
    if args.default_threshold:
        file_name="predictions_default.csv"
    else:
        file_name="predictions_"+str(args.val_min_recall).split(".")[-1]+".csv"
    
    with open(os.path.join(output_dir , file_name), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k, t, m, n, p, q in zip(snapshot_id, thread_id, time, text_length, y_test.values.reshape(-1), y_pred, test_pred, [best_threshold]*len(y_pred)):
            writer.writerow(
                {'snapshot_id':i,'thread_id':j,'time':k,'text_length':t, 'True_label': m , 'Predicted_label': n, 'Predicted_prob': p, 'best_threshold':q})
    
    # file_name=args.model_name+"_"+"metrics_test.txt"
    # output_dir=os.path.join(os.getcwd(),"tfidf+structure")
    # with open(os.path.join(output_dir,file_name),'a') as f:
    #     f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
    #     {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  
        
    # output_name=args.model_name+"_"+"y_true_pred.txt"
    # with open(os.path.join(output_dir,output_name),'w') as f:
    #     for x,y,z in zip(y_test.target_variable.tolist(),y_pred,test_pred.tolist()):
    #         f.write(str(x)+","+str(y)+","+str(z)+ '\n')

    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".\
           format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))


    print()
    print(f"\n===========Test Set Performance===============\n")
    print()
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    print(classification_report(y_test, y_pred))
    print()
    print(confusion_matrix(y_test, y_pred))  
    