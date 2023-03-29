import os
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from arg_parser import arg_parse
import utils

root_dir="/opt/omniai/work/instance1/jupyter"
data_dir=os.path.join(root_dir,"email-complaints","datasets")

# hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))
train_df=load_from_disk(os.path.join(data_dir,"train_df"))

args=arg_parse()
args.undersampling=True
args.train_negative_positive_ratio=3
print()
print(args)
print()
#### only under-sampling email in training dataset
if args.undersampling:
    train_df.set_format(type="pandas")
    train_data=train_df[:]
    df_train=utils.under_sampling(train_data,'target', args.seed, args.train_negative_positive_ratio)

    df_train.reset_index(drop=True, inplace=True)  

    train_df=Dataset.from_pandas(df_train)

train_df.save_to_disk(os.path.join(os.getcwd(),'data_tempt'))

print(train_df)
