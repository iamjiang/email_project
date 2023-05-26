import csv
import os
import time
import datetime
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
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
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_scheduler
)

from accelerate import Accelerator

import utils


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
    
def under_sampling(df,target, seed, negative_positive_ratio):
    np.random.seed(seed)
    LABEL=df[target].values.squeeze()
    feedback=df["is_feedback"].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    positive_idx=IDX[LABEL==1]
    negative_idx_1=np.random.choice(IDX[(LABEL==0) & (feedback=="N")],size=(len(positive_idx)*negative_positive_ratio))
    negative_idx_2=IDX[feedback=="Y"] ### include feedback=1 in the negative sample
    
    _idx=np.concatenate([positive_idx, negative_idx_1, negative_idx_2])

    return df.loc[_idx,:]
    
def main(args,train_data, val_data, device):

    train_df=Dataset.from_pandas(train_data)
    val_df=Dataset.from_pandas(val_data)
   
    if args.customized_model:
        model_path=args.model_name.split("-")[0]+"_"+args.model_name.split("-")[1]+"_"+"repo"
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "v2_new_email/fine-tune-LM",model_path)
    else:
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
        
    config=AutoConfig.from_pretrained(model_name)
    
    tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=args.max_token_length)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    # model=AutoModel.from_pretrained(model_name)
    # self_dropout=nn.Dropout(0.2)
    # self_linear=nn.Linear(config.hidden_size,2)

    if args.frozen_layers!=0:
        modules = [model.base_model.embeddings, *model.base_model.encoder.layer[:args.frozen_layers]] 
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
            
    if args.frozen_all:
        for param in model.parameters():
            param.requires_grad = False
            
    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()
    
    train_module=utils.Loader_Creation(train_df, tokenizer,args.feature_name)
    val_module=utils.Loader_Creation(val_df, tokenizer,args.feature_name)

    train_dataloader=DataLoader(train_module,
                                shuffle=True,
                                batch_size=args.train_batch_size,
                                collate_fn=train_module.collate_fn,
                                drop_last=False,
                                pin_memory=True,
                                # num_workers=4
                               )

    valid_dataloader=DataLoader(val_module,
                                shuffle=False,
                                batch_size=args.val_batch_size,
                                collate_fn=train_module.collate_fn,
                                pin_memory=True,
                                # num_workers=4
                               )

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    
    train_label=train_data['target'].values.squeeze()
    num_classes=np.unique(train_label).shape[0]
    if args.loss_weight:
        train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
        loss_weight=torch.tensor(train_classes_weight).to(device)
    else:
        loss_weight=None
        

    t_total = int((len(train_dataloader) // args.train_batch_size)//args.gradient_accumulation_steps*float(args.num_epochs))

    warmup_steps=int((len(train_dataloader) // args.train_batch_size)//args.gradient_accumulation_steps*args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    # optimizer=AdamW(model.parameters(),lr=args.lr)
    #     lr_scheduler =get_linear_schedule_with_warmup(optimizer, 
    #                                                   num_warmup_steps=warmup_steps, 
    #                                                   num_training_steps=t_total
    #                                                  )

    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=t_total)
    
    accelerator = Accelerator(fp16=args.fp16)
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    if accelerator.is_main_process:
        accelerator.print("")
        logger.info(f'Accelerator Config: {acc_state}')
        accelerator.print("")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable() 
        
    model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader
    )
    
    best_metric = float('inf')
    # best_metric = 0
    best_epoch = 0
    
    iter_tput = []
    
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
        
        accelerator.print(f"\n===========EPOCH {epoch+1}/{args.num_epochs}===============\n")
        model.train()
                
        losses=[]
        for step,batch in enumerate(train_dataloader):
            t0=time.time()
            batch={k:v.type(torch.LongTensor).to(accelerator.device) for k,v in batch.items()}
            inputs={k:v  for k,v in batch.items() if k!="labels"}
            outputs=model(**inputs)
            logits=outputs['logits']
            # logits=outputs['logits']
            
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(accelerator.device),batch["labels"])
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(accelerator.device),batch["labels"], \
                                       weight=loss_weight.float().to(accelerator.device)) 
            
            accelerator.backward(loss)
            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
                optimizer.step()
                if args.use_schedule:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
            losses.append(loss.item())
            
            iter_tput.append(batch["input_ids"].shape[0] / (time.time() - t0))
            
            if step%(len(train_dataloader)//10)==0 and not step==0 :
                accelerator.print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                                  .format(epoch, step, np.mean(losses[-10:]), np.mean(iter_tput[3:]), 
                                          torch.cuda.max_memory_allocated() / 1000000))       
        
        val_pred,val_target,val_losses=utils.eval_func(valid_dataloader,
                                               model, 
                                               accelerator.device,
                                               num_classes=num_classes, 
                                               loss_weight=loss_weight)

 
        output_dir=os.path.join(os.getcwd(), args.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        avg_val_loss=np.mean(val_losses)

        selected_metric=avg_val_loss
        if selected_metric<best_metric:
            best_metric=selected_metric
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
        #Early stopping
        if epoch - best_epoch > args.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, best_metric))
            break 

    config=AutoConfig.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    val_pred,val_target,val_losses=utils.eval_func(valid_dataloader,
                                                   model, 
                                                   accelerator.device,
                                                   num_classes=num_classes, 
                                                   loss_weight=loss_weight)
        

    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), 
                                                min_recall=args.val_min_recall, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in val_pred[:,1]]
    val_output=utils.model_evaluate(val_target.reshape(-1),val_pred[:,1],best_threshold)

    if accelerator.is_main_process:
        with open(os.path.join(output_dir,"metrics_val.txt"),'a') as f:
            f.write(f'{args.model_name},{val_output["total positive"]},{val_output["false positive"]},\
            {val_output["false_negative"]}, {val_output["precision"]},{val_output["recall"]},{val_output["f1_score"]},\
            {val_output["AUC"]},{val_output["pr_auc"]},{best_threshold}\n')   
                
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    # parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")

    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in validation")

    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gradient_checkpointing',  action="store_true")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument('--customized_model',  action="store_true")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--frozen_layers", type=int, default=0,help="freeze layers without gradient updates")
    parser.add_argument("--frozen_all", action='store_true', default=0,help="freeze all parameters")
    parser.add_argument("--hidden_dropout_prob", default=0.2, type=float, help="dropout rate for hidden state.")
    parser.add_argument("--es_patience", type=int, default=3,
                            help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. \
                            Set to 0 to disable this technique.")
    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--max_token_length", type=int, default=512)
    
    args= parser.parse_args()

    if args.customized_model:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_" + args.model_name.split("-")[2] + "_customized"
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_customized"
    else:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]+ "_" + args.model_name.split("-")[2]
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]


    seed_everything(args.seed)

    print()
    print(args)
    print()
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v2_new_email","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))
    
    ### only keep emails with status=closed
    df=df[df.state=="closed"]
    
    ### keep short emails with text_length<=512
    df=df[df.text_length<=512]
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2]) \
    else ("val" if (row["year"]==2023 and row["month"]==3) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    
    df.loc[:,'target']=df.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    train_df=datasets.Dataset.from_pandas(df[df["data_type"]=="train"])
    val_df=datasets.Dataset.from_pandas(df[df["data_type"]=="val"])
    test_df=datasets.Dataset.from_pandas(df[df["data_type"]=="test"])
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id','thread_id','time','is_feedback',\
                                    'preprocessed_email','is_complaint','target',])

    train_df=hf_data["train"]
    val_df=hf_data["validation"]
    test_df=hf_data["test"]
    
    def under_sampling_func(df,feature,negative_positive_ratio,seed=101):
        df.set_format(type="pandas")
        data=df[:]
        df=under_sampling(data,feature, seed, negative_positive_ratio)
        df.reset_index(drop=True, inplace=True)  
        df=datasets.Dataset.from_pandas(df)
        return df
    if args.train_undersampling:
        train_df=under_sampling_func(train_df,"target",args.train_negative_positive_ratio,args.seed)
    if args.val_undersampling:
        val_df=under_sampling_func(val_df,"target",args.val_negative_positive_ratio,args.seed)
        
    hf_data=DatasetDict({"train":train_df, "validation":val_df})
    
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print()
        print('{:<30}{:<10}'.format("The # of availabe GPU(s): ",torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            print('{:<30}{:<10}'.format("GPU Name: ",torch.cuda.get_device_name(i)))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    train_data=hf_data['train'].shuffle(seed=101).select(range(len(hf_data["train"])))
    train_data.set_format(type="pandas")
    train_data=train_data[:]
    # train_data=train_data.sample(500)
    
    val_data=hf_data['validation'].shuffle(seed=101).select(range(len(hf_data["validation"])))
    val_data.set_format(type="pandas")
    val_data=val_data[:]
    # val_data=val_data.sample(500)
    
    main(args,train_data, val_data, device)
    