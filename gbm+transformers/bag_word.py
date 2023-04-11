import time
import os
import re
import operator
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

    # text = [appos[word] if word in appos else word for word in text.lower().split()]
    # text = " ".join(text)
    
    ### Remove stop word
    text = [word for word in word_tokenize(text) if word.lower() not in STOPWORDS]
    text = " ".join(text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    #Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split()]
    text=" ".join(text)
    
    # stem
    # ps = PorterStemmer()
    # text=" ".join(set([ps.stem(w) for w in text.split()]))
    
    return text

root_dir="/opt/omniai/work/instance1/jupyter"
data_dir=os.path.join(root_dir,"email-complaints","datasets")
hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))

train_df=hf_data["train"]
train_df.set_format(type="pandas")
df_train=train_df[:]
    
val_df=hf_data["validation"]
val_df.set_format(type="pandas")
df_val=val_df[:]

test_df=hf_data["test"]
test_df.set_format(type="pandas")
df_test=test_df[:]
    
df_train["bag_of_word"]=df_train["preprocessed_email"].progress_apply(text_preprocess)
df_val["bag_of_word"]=df_val["preprocessed_email"].progress_apply(text_preprocess)
df_test["bag_of_word"]=df_test["preprocessed_email"].progress_apply(text_preprocess)

# df_train["adj_bag_of_word"]=df_train["preprocessed_email"].progress_apply(lambda x: text_preprocess(x, extract_adj=True))
# df_val["adj_bag_of_word"]=df_val["preprocessed_email"].progress_apply(lambda x: text_preprocess(x, extract_adj=True))
# df_test["adj_bag_of_word"]=df_test["preprocessed_email"].progress_apply(lambda x: text_preprocess(x, extract_adj=True))\

## removing non-english words from text
words = set(nltk.corpus.words.words())

df_train["bag_of_word"] = df_train["bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
df_val["bag_of_word"] = df_val["bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
df_test["bag_of_word"] = df_test["bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))

# df_train["adj_bag_of_word"] = df_train["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
# df_val["adj_bag_of_word"] = df_val["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
# df_test["adj_bag_of_word"] = df_test["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))

## removing short text
# df_train["adj_bag_of_word"] = df_train["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3) )
# df_val["adj_bag_of_word"] = df_val["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3))
# df_test["adj_bag_of_word"] = df_test["adj_bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3))

# df_train["bag_of_word"] = df_train["bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3))
# df_val["bag_of_word"] = df_val["bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3))
# df_test["bag_of_word"] = df_test["bag_of_word"].progress_apply(lambda x: " ".join(w for w in x.split() if len(w)>=3))


df_train.to_pickle(os.path.join(os.getcwd(),"df_train"))
df_val.to_pickle(os.path.join(os.getcwd(),"df_val"))
df_test.to_pickle(os.path.join(os.getcwd(),"df_test"))

# df_train=pd.read_pickle(os.path.join(os.getcwd(),"df_train"))
# df_val=pd.read_pickle(os.path.join(os.getcwd(),"df_val"))
# df_test=pd.read_pickle(os.path.join(os.getcwd(),"df_test"))

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

# df_train['negative_word_counts'] = 0
# df_val['negative_word_counts'] = 0
# df_test['negative_word_counts'] = 0
# for w in tqdm(negative_word, total=len(negative_word)):
#     df_train['negative_word_counts']+=df_train["bag_of_word"].apply(lambda x: w in x)
#     df_val['negative_word_counts']+=df_val["bag_of_word"].apply(lambda x: w in x)
#     df_test['negative_word_counts']+=df_test["bag_of_word"].apply(lambda x: w in x)

# def negative_keyword_complaint_ratio(df,num_neg):
#     tempt=df[df['negative_word_counts']<=num_neg]
#     ratio=tempt[tempt["is_complaint"]=="Y"].shape[0]/tempt.shape[0]
#     return (num_neg,ratio)

# train_neg_keyword_complaint=[]
# val_neg_keyword_complaint=[]
# test_neg_keyword_complaint=[]
# max_num=min(df_train['negative_word_counts'].max(), df_val['negative_word_counts'].max(), df_test['negative_word_counts'].max())
# for i in range(0,max_num,5):
#     train_neg_keyword_complaint.append(negative_keyword_complaint_ratio(df_train,i))
#     val_neg_keyword_complaint.append(negative_keyword_complaint_ratio(df_val,i))
#     test_neg_keyword_complaint.append(negative_keyword_complaint_ratio(df_test,i))
    

# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rc
# sns.set(style="whitegrid",palette='muted',font_scale=1.2)
# # rcParams['figure.figsize']=16,10
# %config InlineBackend.figure_format="retina"
# %matplotlib inline

# plt.rcParams["figure.figsize"] = (15,8)
# plt.plot([int(x[0]) for x in train_neg_keyword_complaint],  [x[1] for x in train_neg_keyword_complaint], label = "Training",color='k',marker='h', linestyle="-.", linewidth=3)
# plt.plot([int(x[0]) for x in val_neg_keyword_complaint],  [x[1] for x in val_neg_keyword_complaint], label = "Validation",color='b',marker='d', linestyle="--", linewidth=3)
# plt.plot([int(x[0]) for x in test_neg_keyword_complaint],  [x[1] for x in test_neg_keyword_complaint], label = "Test",color='r',marker='v', linestyle=":", linewidth=3)
# # plt.axvline(x=8, color='b')
# plt.legend(fontsize="x-large")
# plt.title("complaint vs number of negative keyword",fontsize=20)
# plt.ylabel("complaint",fontsize=15)
# plt.xlabel("number of negative sentiment words in email",fontsize=12)
# plt.xticks(rotation=90)
# plt.xticks(np.arange(0, max_num, 5))
# plt.show()

# tempt1=df_train.copy()
# tempt1["data_type"]=["training_set"]*len(tempt1)
# tempt2=df_test.copy()
# tempt2["data_type"]=["test_set"]*len(tempt2)
# tempt=pd.concat([tempt1,tempt2],axis=0)

# word=["frustration","frustrated","frustrate","unacceptable","disappointed","disappointing","unhappy","misunderstanding"]
# tempt["negative_word"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(word).intersection(set(x.split())))!=0 else 0 )

# plt.rcParams["figure.figsize"] = [10, 5]
# plt.rcParams["figure.autolayout"] = True
# ax = sns.barplot(data = tempt, x='negative_word',y='target',hue="data_type")
# ax.set_title(" selected negative word existing in email")
# plt.legend(loc="best")


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

# keyword_training=[w for w in word_train_complaint if w not in word_train_no_churn]
# keyword_test=[w for w in word_test_complaint if w not in word_test_no_churn]

dict_data={}
dict_data["training"]=word_train_complaint[0:50]
dict_data["validation"]=word_val_complaint[0:50]
dict_data["test"]=word_test_complaint[0:50]
pd.DataFrame(dict_data).style.format().set_caption("Most common negative sentiment word in complaint==1")\
.set_table_styles([{'selector': 'caption','props': [('color', 'red'),('font-size', '15px')]}]) 

# word=["frustration","frustrated","frustrate","unacceptable","disappointed","disappointing","unhappy","misunderstanding"]
word=set(word_train_complaint[0:20]).difference(set(word_train_no_ccomplaint[0:50]))
print()
print(word)
print()
# tempt["negative_word"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(word).intersection(set(x.split())))!=0 else 0 )
df_train["negative_word"]=df_train["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
df_val["negative_word"]=df_val["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )
df_test["negative_word"]=df_test["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0 )

tempt1=df_train.copy()
tempt1["data_type"]=["training_set"]*len(tempt1)
tempt2=df_val.copy()
tempt2["data_type"]=["validation_set"]*len(tempt2)
tempt3=df_test.copy()
tempt3["data_type"]=["test_set"]*len(tempt3)
tempt=pd.concat([tempt1,tempt2,tempt3],axis=0)

plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
ax = sns.barplot(data = tempt, x='negative_word',y='target',hue="data_type")
ax.set_title(" selected negative word existing in email")
plt.legend(loc="best")


#####################################################################################################################

tempt=df_train[df_train["target"]==0]
tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if (len(x.split())>0 and any(item in word for item in x.split())) else 0)
# tempt["not_complaint_negative"]=tempt["bag_of_word"].progress_apply(lambda x: 1 if len(set(('disappointed', 'unacceptable')).intersection(x))>0 else 0)
tempt["not_complaint_negative"].value_counts(dropna=False)

tempt=tempt[tempt["not_complaint_negative"]==1]
tempt[tempt.text_length<200].shape
tempt=tempt[tempt.text_length<200]

import textwrap
import random

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=150) 

exam_text = tempt.loc[:,["snapshot_id","preprocessed_email","is_complaint"]]

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
    print(exam_text.loc[j,"is_complaint"])
    print()
    print(wrapper.fill(exam_text.loc[j,"preprocessed_email"]))
    

    #########################################################################################

# def pcut_func(df,var,nbin=10):
#     df[var]=df[var].astype(float)
#     df["text_length_decile"]=pd.qcut(df[var],nbin,precision=2,duplicates="drop")
#     # df.drop(var, axis=1,inplace=True)
#     return df

# df_train=pcut_func(df_train,var="text_length",nbin=10)
# df_val=pcut_func(df_val,var="text_length",nbin=10)
# df_test=pcut_func(df_test,var="text_length",nbin=10)

def refine_data(df):
    x=df.loc[:,["service","front office","negative_word","text_length"]]
    y=df.loc[:,["target"]]
    x["service"]=x["service"].astype(str)
    x["front office"]=x["front office"].astype(str)
    x["negative_word"]=x["negative_word"].astype(str)
    # x["text_length_decile"]=x["text_length_decile"].astype(str)
    return x,y

x_train,y_train=refine_data(df_train)
x_val,y_val=refine_data(df_val)
x_test,y_test=refine_data(df_test)

# cat_features_names = [col for col in x_train.columns if x_train[col].dtypes=="object"]
cat_features_names = ["service","front office","negative_word"]
cat_features = [x_train.columns.get_loc(col) for col in cat_features_names]

train_data = Pool(data=x_train,
                  label=y_train,
                  cat_features=cat_features
                 )

val_data = Pool(data=x_val,
                  label=y_val,
                  cat_features=cat_features
                 )

test_data = Pool(data=x_test,
                  label=y_test,
                  cat_features=cat_features
                 )

train_label=y_train['target'].values.squeeze()
num_classes=np.unique(train_label).shape[0]
train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
loss_weight=train_classes_weight

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'iterations': 1000,
          'learning_rate': 0.05,
#           'cat_features': cat_features, # we don't need to specify this parameter as 
#                                           pool object contains info about categorical features
          'early_stopping_rounds': 50,
          'verbose': 200,
          'random_seed': 101,
          'scale_pos_weight': loss_weight[1]/loss_weight[0]
         }

clf = CatBoostClassifier(**params)
clf.fit(train_data, # instead of X_train, y_train
          eval_set=val_data, # instead of (X_valid, y_valid)
          use_best_model=True, 
          plot=True
         );


train_pred=clf.predict_proba(x_train)[:,1]
val_pred=clf.predict_proba(x_val)[:,1]
test_pred=clf.predict_proba(x_test)[:,1]

best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=0.85, pos_label=False)
y_pred=[1 if x>best_threshold else 0 for x in test_pred]
test_output=utils.model_evaluate(y_test.values.reshape(-1),y_pred)

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

# clf.get_feature_importance(prettified=True)
feature_importance_df = pd.DataFrame(clf.get_feature_importance(prettified=True))
plt.figure(figsize=(12, 6));
sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
plt.title('CatBoost features importance:');

clf.get_feature_importance(prettified=True)


#################### TFIDF  ###############################


# bow_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=7000, stop_words='english')
# # bag-of-words feature matrix
# bow = bow_vectorizer.fit_transform(df_train['bag_of_word'])

# top_sum=bow.toarray().sum(axis=0)
# top_sum_cv=[top_sum]
# # columns_cv = bow_vectorizer.get_feature_names()
# columns_cv = bow_vectorizer.vocabulary_.keys()
# x_traincvdf = pd.DataFrame(top_sum_cv,columns=columns_cv)

# dic = {}
# for i in range(len(top_sum_cv[0])):
#     dic[list(columns_cv)[i]]=top_sum_cv[0][i]
# sorted_dic=sorted(dic.items(),reverse=True,key=operator.itemgetter(1))
# print(sorted_dic[0:100])


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
bow_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=2, max_features=7000, stop_words='english')
# bag-of-words feature matrix
# bow = bow_vectorizer.fit_transform(df_train['preprocessed_email'])
bow = bow_vectorizer.fit_transform(df_train['bag_of_word'])
train_tfidf = bow.toarray()
vocab = bow_vectorizer.vocabulary_.keys()
vocab = list(vocab)
train_tfidf = pd.DataFrame(train_tfidf,columns=vocab)

# val_tfidf = bow_vectorizer.transform(df_val['preprocessed_email'])
val_tfidf = bow_vectorizer.transform(df_val['bag_of_word'])
val_tfidf = val_tfidf.toarray()
val_tfidf = pd.DataFrame(val_tfidf,columns=vocab)

# test_tfidf = bow_vectorizer.transform(df_test['preprocessed_email'])
test_tfidf = bow_vectorizer.transform(df_test['bag_of_word'])
test_tfidf = test_tfidf.toarray()
test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)

from catboost import CatBoostClassifier, Pool

# cat_features_names = [col for col in train_tfidf.columns if train_tfidf[col].values.dtypes=="object"]
# cat_features = [train_tfidf.columns.get_loc(col) for col in cat_features_names]

cat_features = []

y_train=df_train.loc[:,["target"]]
y_val=df_val.loc[:,["target"]]
y_test=df_test.loc[:,["target"]]

train_data = Pool(data=train_tfidf,
                  label=y_train,
                  cat_features=cat_features
                 )

val_data = Pool(data=val_tfidf,
                  label=y_val,
                  cat_features=cat_features
                 )

test_data = Pool(data=test_tfidf,
                  label=y_test,
                  cat_features=cat_features
                 )

train_label=y_train['target'].values.squeeze()
num_classes=np.unique(train_label).shape[0]
train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
loss_weight=train_classes_weight

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'iterations': 1000,
          'learning_rate': 2e-5,
#           'cat_features': cat_features, # we don't need to specify this parameter as 
#                                           pool object contains info about categorical features
          'early_stopping_rounds': 50,
          'verbose': 200,
          'random_seed': 101,
          'scale_pos_weight': loss_weight[1]/loss_weight[0]
         }

clf = CatBoostClassifier(**params)
clf.fit(train_data, # instead of X_train, y_train
          eval_set=val_data, # instead of (X_valid, y_valid)
          use_best_model=True, 
          plot=True
         );


train_pred=clf.predict_proba(train_tfidf)[:,1]
val_pred=clf.predict_proba(val_tfidf)[:,1]
test_pred=clf.predict_proba(test_tfidf)[:,1]

# best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze())
best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=0.85, pos_label=False)
y_pred=[1 if x>best_threshold else 0 for x in test_pred]
test_output=utils.model_evaluate(y_test.values.reshape(-1),y_pred)

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

# clf.get_feature_importance(prettified=True)
# feature_importance_df = pd.DataFrame(clf.get_feature_importance(prettified=True))
# plt.figure(figsize=(12, 6));
# sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
# plt.title('CatBoost features importance:');

# clf.get_feature_importance(prettified=True)


#################### TFIDF + Structure Data ###############################

# def pcut_func(df,var,nbin=10):
#     df[var]=df[var].astype(float)
#     df["text_length_decile"]=pd.qcut(df[var],nbin,precision=2,duplicates="drop")
#     # df.drop(var, axis=1,inplace=True)
#     return df

# df_train=pcut_func(df_train,var="text_length",nbin=10)
# df_val=pcut_func(df_val,var="text_length",nbin=10)
# df_test=pcut_func(df_test,var="text_length",nbin=10)

def refine_data(df):
    x=df.loc[:,["service","front office","negative_word","text_length"]]
    y=df.loc[:,["target"]]
    x["service"]=x["service"].astype(str)
    x["front office"]=x["front office"].astype(str)
    x["negative_word"]=x["negative_word"].astype(str)
    # x["text_length_decile"]=x["text_length_decile"].astype(str)
    return x,y

x_train,y_train=refine_data(df_train)
x_val,y_val=refine_data(df_val)
x_test,y_test=refine_data(df_test)

X_Train=pd.concat([train_tfidf.reset_index(drop=True),x_train.reset_index(drop=True)],axis=1)
X_Val=pd.concat([val_tfidf.reset_index(drop=True),x_val.reset_index(drop=True)],axis=1)
X_Test=pd.concat([test_tfidf.reset_index(drop=True),x_test.reset_index(drop=True)],axis=1)


# cat_features_names = [col for col in X_Train.columns if X_Train[col].dtypes=="object"]
# cat_features = [X_Train.columns.get_loc(col) for col in cat_features_names]

# cat_features_names = [col for col in x_train.columns if x_train[col].dtypes=="object"]
cat_features_names = ["service","front office","negative_word"]
cat_features = [X_Train.columns.get_loc(col) for col in cat_features_names]

train_data = Pool(data=X_Train,
                  label=y_train,
                  cat_features=cat_features
                 )

val_data = Pool(data=X_Val,
                  label=y_val,
                  cat_features=cat_features
                 )

test_data = Pool(data=X_Test,
                  label=y_test,
                  cat_features=cat_features
                 )

train_label=y_train['target'].values.squeeze()
num_classes=np.unique(train_label).shape[0]
train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
loss_weight=train_classes_weight

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'iterations': 1000,
          'learning_rate': 2e-5,
#           'cat_features': cat_features, # we don't need to specify this parameter as 
#                                           pool object contains info about categorical features
          'early_stopping_rounds': 50,
          'verbose': 200,
          'random_seed': 101,
          'scale_pos_weight': loss_weight[1]/loss_weight[0]
         }

clf = CatBoostClassifier(**params)
clf.fit(train_data, # instead of X_train, y_train
          eval_set=val_data, # instead of (X_valid, y_valid)
          use_best_model=True, 
          plot=True
         );


train_pred=clf.predict_proba(X_Train)[:,1]
val_pred=clf.predict_proba(X_Val)[:,1]
test_pred=clf.predict_proba(X_Test)[:,1]

# best_threshold=find_optimal_threshold(y_val.squeeze(), val_pred.squeeze())
best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=0.85, pos_label=False)
y_pred=[1 if x>best_threshold else 0 for x in test_pred]
test_output=utils.model_evaluate(y_test.values.reshape(-1),y_pred)

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

# clf.get_feature_importance(prettified=True)
# feature_importance_df = pd.DataFrame(clf.get_feature_importance(prettified=True))
# plt.figure(figsize=(12, 6));
# sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
# plt.title('CatBoost features importance:');

# clf.get_feature_importance(prettified=True)
