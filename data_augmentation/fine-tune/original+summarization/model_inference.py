import csv
import os
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
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from accelerate import Accelerator

import utils

from torch.utils.data import DataLoader, TensorDataset

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
    
def main(args,val_data, test_data, device):

    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    

    model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","data_augmentation/fine-tune/original-data",args.output_dir)
    config=AutoConfig.from_pretrained(model_name)
    if args.model_name=="bigbird-roberta-large":
        # config.block_size=32
        # config.num_random_blocks=2
        config.attention_type="original_full"
        tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=1500)
        model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings-2)
        model=AutoModelForSequenceClassification.from_pretrained(model_name)
          
    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()
    
    val_module=utils.Loader_Creation(val_df, tokenizer,args.feature_name)
    test_module=utils.Loader_Creation(test_df, tokenizer,args.feature_name)

    valid_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=val_module.collate_fn
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=test_module.collate_fn
                               )

    print()
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    
    def eval_func(data_loader,model,device,num_classes=2):
        
        fin_targets=[]
        fin_outputs=[]
        losses=[]
        model.eval()
        if args.multiple_gpus:
            model=model.to(device[0])
            model=torch.nn.DataParallel(model,device_ids=device)
        else:
            model=model.to(device)

        batch_idx=0
        for batch in tqdm(data_loader, position=0, leave=True):
            if not args.multiple_gpus:
                batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            else:
                batch={k:v.type(torch.LongTensor).to(device[0]) for k,v in batch.items()}
                
            inputs={k:v  for k,v in batch.items() if k!="labels"}
            with torch.no_grad():
                outputs=model(**inputs)
            logits=outputs['logits']

            loss = F.cross_entropy(logits.view(-1, num_classes), batch["labels"])
   
            losses.append(loss.item())

            fin_targets.append(batch["labels"].cpu().detach().numpy())
            fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())   

            batch_idx+=1
            
        return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses 

    val_pred,val_target,val_losses=eval_func(valid_dataloader,
                                             model, 
                                             device)
        
    test_pred,test_target,test_losses=eval_func(test_dataloader,
                                                model, 
                                                device)

    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)

    output_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","data_augmentation/fine-tune/original-data",args.output_dir)

    with open(os.path.join(output_dir,"metrics_test.txt"),'a') as f:
        f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
        {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  

    fieldnames = ['True label', 'Predicted label', 'Predicted_prob']
    with open(os.path.join(output_dir,"predictions.csv"),'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(test_target, y_pred, test_pred[:,1]):
            writer.writerow(
                {'True label': i, 'Predicted label': j, 'Predicted_prob': k})  


    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))

    print()
    print(f"\n===========Test Set Performance===============\n")
    print()

    print(classification_report(test_target, y_pred))
    print()
    print(confusion_matrix(test_target, y_pred))  
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--multiple_gpus', action="store_true", help="use multiple gpus or not")
    parser.add_argument('--gpus', type=int, default=[], nargs='+', help='used gpu')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--data_augmentation", type=str, default="summarization")
    args= parser.parse_args()

    args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] 

    seed_everything(args.seed)

    print()
    print(args)
    print()
    

    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","datasets")
    argment_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","data_augmentation/augmented_data")
    
    train_val_test=pd.read_pickle(os.path.join(data_dir,"new_train_val_test_pickle"))
    train_df1=train_val_test[train_val_test["data_type"]=="training_set"].loc[:,['snapshot_id', 'preprocessed_email','is_complaint']]
    if args.data_augmentation=="summarization":
        train_augment=pd.read_pickle(os.path.join(argment_dir,"summarized_email_pickle"))
    elif args.data_augmentation=="back-translation":
        train_augment=pd.read_pickle(os.path.join(argment_dir,"back_translation_pickle"))
    elif args.data_augmentation=="masked-language":
        train_augment=pd.read_pickle(os.path.join(argment_dir,"masked_email_pickle"))
    elif args.data_augmentation=="all":
        train_augment_v1=pd.read_pickle(os.path.join(argment_dir,"summarized_email_pickle"))
        train_augment_v2=pd.read_pickle(os.path.join(argment_dir,"back_translation_pickle"))
        train_augment_v3=pd.read_pickle(os.path.join(argment_dir,"masked_email_pickle"))
        train_augment=pd.concat([train_augment_v1,train_augment_v2,train_augment_v3])
        
    train_df2=train_augment.loc[:,['snapshot_id', 'preprocessed_email','is_complaint']]
    train_df=pd.concat([train_df1,train_df2])
    
    val_df=train_val_test[train_val_test["data_type"]=="validation_set"].loc[:,['snapshot_id', 'preprocessed_email','is_complaint']]
    test_df=train_val_test[train_val_test["data_type"]=="test_set"].loc[:,['snapshot_id', 'preprocessed_email','is_complaint']]
    
    train_df=datasets.Dataset.from_pandas(train_df)
    val_df=datasets.Dataset.from_pandas(val_df)
    test_df=datasets.Dataset.from_pandas(test_df)
    
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])
    
    def binary_label(example):
        return {"target": 1 if example["is_complaint"]=="Y" else 0}

    train_df=hf_data["train"].map(binary_label)
    val_df=hf_data["validation"].map(binary_label)
    test_df=hf_data["test"].map(binary_label)
        
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    
    if args.multiple_gpus:
        device=[torch.device(f"cuda:{i}") for i in args.gpus]
    else:
        device=torch.device(args.device)
    
    val_data=hf_data['validation'].shuffle(seed=101).select(range(len(hf_data["validation"])))
    val_data.set_format(type="pandas")
    val_data=val_data[:]
    # val_data=val_data.sample(500)
    
    test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    test_data.set_format(type="pandas")
    test_data=test_data[:]
    # test_data=test_data.sample(500)
    
    main(args,val_data, test_data, device)
    