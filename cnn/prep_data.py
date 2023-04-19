import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
all_stopwords_gensim = STOPWORDS.union(set(['thank','thanks', 'you', 
                                        'help','questions','a.m.','p.m.','friday','thursday','wednesday','tuesday','monday','appreciate','available',\
                                            'hello','hi','?','.','. .','phone','needs','need','let','know','service','information','time','meet']))

stopwords=set(all_stopwords_gensim)
stopwords.discard("not")


input_dir="/opt/omniai/work/instance1/jupyter/email-complaints/datasets"
df=pd.read_pickle(os.path.join(input_dir,"train_val_test_pickle"))

train_df=df[df["data_type"]=="training_set"].loc[:,["preprocessed_email","is_complaint"]]
val_df=df[df["data_type"]=="validation_set"].loc[:,["preprocessed_email","is_complaint"]]
test_df=df[df["data_type"]=="test_set"].loc[:,["preprocessed_email","is_complaint"]]

train_df["target"]=train_df["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)
train_df=train_df.loc[:,["preprocessed_email","target"]]
train_df["preprocessed_email"] = train_df["preprocessed_email"].progress_apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
train_df.to_csv("train_df.csv")

val_df["target"]=val_df["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)
val_df=val_df.loc[:,["preprocessed_email","target"]]
val_df["preprocessed_email"] = val_df["preprocessed_email"].progress_apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
val_df.to_csv("val_df.csv")

test_df["target"]=test_df["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)
test_df=test_df.loc[:,["preprocessed_email","target"]]
test_df["preprocessed_email"] = test_df["preprocessed_email"].progress_apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df.to_csv("test_df.csv")

