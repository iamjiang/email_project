import os
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
disable_caching()

from arg_parser import arg_parse
import utils

root_dir="/opt/omniai/work/instance1/jupyter"
data_dir=os.path.join(root_dir,"email-complaints","datasets")

# hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))
train_df=load_from_disk(os.path.join(data_dir,"train_df"))

args=arg_parse()
args.undersampling=True
args.train_negative_positive_ratio=5
print()
print(args)
print()

root_dir="/opt/omniai/work/instance1/jupyter"
data_dir=os.path.join(root_dir,"email-complaints","datasets")
    
train_df=load_from_disk(os.path.join(data_dir,"train_df"))
val_df=load_from_disk(os.path.join(data_dir,"val_df"))
test_df=load_from_disk(os.path.join(data_dir,"test_df"))

train_df=train_df.filter(lambda x: x['text_length']>10)
val_df=val_df.filter(lambda x: x['text_length']>10)
test_df=test_df.filter(lambda x: x['text_length']>10)

#### only under-sampling email in training dataset
# if args.undersampling:
#     train_df.set_format(type="pandas")
#     train_data=train_df[:]
#     df_train=utils.under_sampling(train_data,'target', args.seed, args.train_negative_positive_ratio)

#     df_train.reset_index(drop=True, inplace=True)  

#     train_df=Dataset.from_pandas(df_train)

hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})

#     model_checkpoint=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_checkpoint)
#     config=AutoConfig.from_pretrained(model_checkpoint)
#     tokenizer=AutoTokenizer.from_pretrained(model_checkpoint,model_max_length=config.max_position_embeddings-2)
#     max_seq_length=config.max_position_embeddings-2
#     def truncation_text(example):
#         truncated_input_ids=tokenizer(example[args.feature_name],truncation=True,padding=False,return_tensors="pt",add_special_tokens=False)['input_ids']

#         if args.truncation_strategy=="tail":
#             truncated_input_ids=truncated_input_ids[:,-(max_seq_length - 2):].squeeze()
#         elif args.truncation_strategy=="head":
#             truncated_input_ids=truncated_input_ids[:,0:(max_seq_length - 2)].squeeze()
#         elif args.truncation_strategy=="mixed":
#             truncated_input_ids=truncated_input_ids[:(max_seq_length - 2) // 2] + truncated_input_ids[-((max_seq_length - 2) // 2):]
#             truncated_input_ids=truncated_input_ids.squeeze()
#         else:
#             raise NotImplemented("Unknown truncation. Supported truncation: tail, head, mixed truncation")

#         return {"truncated_text":tokenizer.decode(truncated_input_ids)}

#     hf_data=hf_data.map(truncation_text)
#     columns=hf_data['train'].column_names
#     columns_to_keep=['preprocessed_email','target']
#     columns_to_remove=set(columns)-set(columns_to_keep)
#     hf_data=hf_data.remove_columns(columns_to_remove)

# train_data=hf_data['train'].shuffle(seed=101).select(range(len(hf_data["train"])))
# val_data=hf_data['validation'].shuffle(seed=101).select(range(len(hf_data["validation"])))
# test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))

hf_data.save_to_disk(os.path.join(os.getcwd(),'hf_data'))



# #### only under-sampling email in training dataset
# if args.undersampling:
#     train_df.set_format(type="pandas")
#     train_data=train_df[:]
#     df_train=utils.under_sampling(train_data,'target', args.seed, args.train_negative_positive_ratio)

#     df_train.reset_index(drop=True, inplace=True)  

#     train_df=Dataset.from_pandas(df_train)

# train_df.save_to_disk(os.path.join(os.getcwd(),'data_tempt'))

# print(train_df)
