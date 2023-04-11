import os
import time
import datetime
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import argparse
import logging

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

import transformers
print("Transformers version is {}".format(transformers.__version__))

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

import joblib

import utils
from model_class import ModelForSequenceClassification

import lightgbm as lgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main(args,train_data, val_data, test_data, device):

    train_df=Dataset.from_pandas(train_data)
    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    
    model_checkpoint=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_checkpoint)
    config=AutoConfig.from_pretrained(model_checkpoint)
    tokenizer=AutoTokenizer.from_pretrained(model_checkpoint,model_max_length=config.max_position_embeddings-2)
    model=AutoModel.from_pretrained(model_checkpoint)
#     self_dropout=nn.Dropout(0.2)
#     self_linear=nn.Linear(config.hidden_size,2)

    
#     if args.frozen_layers!=0:
#         modules = [model.base_model.embeddings, *model.base_model.encoder.layer[:args.frozen_layers]] 
#         for module in modules:
#             for param in module.parameters():
#                 param.requires_grad = False
            
#     if args.frozen_all:
#         for param in model.parameters():
#             param.requires_grad = False
            
#     print()
#     print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
#     print(f"Vocabulary size : {tokenizer.vocab_size:,}")
#     print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
#     print()

    train_module=utils.Loader_Creation(train_df, tokenizer,args.feature_name)
    val_module=utils.Loader_Creation(val_df, tokenizer,args.feature_name)
    test_module=utils.Loader_Creation(test_df, tokenizer,args.feature_name)


    train_dataloader=DataLoader(train_module,
                                shuffle=True,
                                batch_size=args.train_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False   # longformer model bug
                               )

    valid_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.test_batch_size,
                                collate_fn=train_module.collate_fn
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.test_batch_size,
                                collate_fn=test_module.collate_fn
                               )

    # %pdb
    # next(iter(train_dataloader))

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    
    train_label=train_data['target'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
    
    
    def create_gb_dataset(dataloader,model):
        # Initialize LightGBM dataset
        lgb_features = None
        lgb_labels=[]
        
        dropout = nn.Dropout(args.hidden_dropout_prob)
        
        for batch in tqdm(dataloader, total=len(dataloader), leave=True, position=0):
            batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            inputs={k:v  for k,v in batch.items() if k!="labels"}
            model=model.to(device)
            model.eval()
            with torch.no_grad():
                outputs=model(**inputs)
                pooled_output=dropout(outputs.pooler_output)
                pooled_output = pooled_output.cpu().detach().numpy()
                y_batch=batch["labels"].cpu().detach().numpy()
            # Concatenate pooled output to LightGBM dataset
            if lgb_features is None:
                lgb_features = pooled_output
            else:
                lgb_features=np.concatenate((lgb_features, pooled_output),axis=0)

            lgb_labels.extend(y_batch)

        lgb_labels=np.array(lgb_labels)

        lgb_data=lgb.Dataset(lgb_features, label=lgb_labels)
        
        return lgb_data, lgb_features, lgb_labels
    
    lgb_train,train_features, train_labels=create_gb_dataset(train_dataloader,model)
    lgb_val,val_features, val_labels=create_gb_dataset(valid_dataloader,model)
    _, test_features, test_labels=create_gb_dataset(test_dataloader,model)
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "verbose": 0,
        "num_threads": 4,
        "scale_pos_weight": loss_weight[1]/loss_weight[0]
    }
    
    # Train LightGBM model on concatenated dataset
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.num_epochs,
        valid_sets=[lgb_val],
        early_stopping_rounds=args.early_stopping_rounds,
    )
    
    if not os.path.exists(os.path.join(os.getcwd(),args.model_output_name)):
        os.makedirs(os.path.join(os.getcwd(),args.model_output_name))
    joblib.dump(lgb_model, os.path.join(os.getcwd(),args.model_output_name,'lgb.pkl'))
    lgb_model = joblib.load(os.path.join(os.getcwd(),args.model_output_name,'lgb.pkl'))
    
    train_pred=lgb_model.predict(train_features)
    val_pred=lgb_model.predict(val_features)
    test_pred=lgb_model.predict(test_features)
    
    # best_threshold=utils.find_optimal_threshold(train_labels, train_pred)
    best_threshold=utils.find_optimal_threshold(val_labels.squeeze(), val_pred.squeeze(), min_recall=0.80, pos_label=False)
    
    
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    test_output=utils.model_evaluate(test_labels.reshape(-1),y_pred)
    
    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".\
           format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))
    
    
    print()
    print(f"\n===========Test Set Performance===============\n")
    print()
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    print(classification_report(test_labels, y_pred))
    print()
    print(confusion_matrix(test_labels, y_pred))  
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')

    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_test_undersampling", action="store_true", help="undersampling or not")

    parser.add_argument("--train_negative_positive_ratio",  type=int,default=2,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--feature_fraction', type=float, default=0.9)
    parser.add_argument('--bagging_fraction', type=float, default=0.8)
    parser.add_argument('--bagging_freq', type=int, default=5)

    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help="early stop rounds for lightgbm")
    
    parser.add_argument('--model_checkpoint', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--model_output_name",  type=str, default=None)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--hidden_dropout_prob", default=0.2, type=float, help="dropout rate for hidden state.")
    

    args= parser.parse_args()

    args.model_output_name=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1]

    seed_everything(args.seed)

    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter"
    data_dir=os.path.join(root_dir,"email-complaints","datasets")
    
#     train_df=load_from_disk(os.path.join(data_dir,"train_df"))
#     val_df=load_from_disk(os.path.join(data_dir,"val_df"))
#     test_df=load_from_disk(os.path.join(data_dir,"test_df"))
    
#     train_df=train_df.filter(lambda x: x['text_length']>10)
#     val_df=val_df.filter(lambda x: x['text_length']>10)
#     test_df=test_df.filter(lambda x: x['text_length']>10)
    
#     hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
#     hf_data.save_to_disk(os.path.join(data_dir,'hf_data'))
#     print(hf_data)
    hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))
    
    train_df=hf_data["train"]
    val_df=hf_data["validation"]
    test_df=hf_data["test"]
    
    def under_sampling_func(df,feature,negative_positive_ratio,seed=101):
        df.set_format(type="pandas")
        data=df[:]
        df=utils.under_sampling(data,feature, seed, negative_positive_ratio)
        df.reset_index(drop=True, inplace=True)  
        df=datasets.Dataset.from_pandas(df)
        return df
    if args.train_undersampling:
        train_df=under_sampling_func(train_df,"target",args.train_negative_positive_ratio,args.seed)
    if args.val_test_undersampling:
        val_df=under_sampling_func(val_df,"target",args.test_negative_positive_ratio,args.seed)
        test_df=under_sampling_func(test_df,"target",args.test_negative_positive_ratio,args.seed)
        
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
    
    train_data=hf_data['train'].shuffle(seed=101).select(range(len(hf_data["train"])))
    val_data=hf_data['validation'].shuffle(seed=101).select(range(len(hf_data["validation"])))
    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    
#     hf_data.save_to_disk(os.path.join(data_dir,'hf_data'))
#     print(hf_data)

#     hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)
    # print(f"The number of GPUs is {torch.cuda.device_count()}")
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print()
        print('{:<30}{:<10}'.format("The # of availabe GPU(s): ",torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            print('{:<30}{:<10}'.format("GPU Name: ",torch.cuda.get_device_name(i)))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    train_data=hf_data['train']
    train_data.set_format(type="pandas")
    train_data=train_data[:]
    # train_data=train_data.sample(500)
    
    val_data=hf_data['validation']
    val_data.set_format(type="pandas")
    val_data=val_data[:]
    # val_data=val_data.sample(500)
    
    test_data=hf_data['test']
    test_data.set_format(type="pandas")
    test_data=test_data[:]
    # test_data=test_data.sample(500)
    
    main(args,train_data, val_data, test_data, device)
    