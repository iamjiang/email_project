import csv
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
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
    get_scheduler
)

from accelerate import Accelerator

import utils

import bitsandbytes as bnb

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
   
    if args.customized_model:
        model_path=args.model_name.split("-")[0]+"_"+args.model_name.split("-")[1]+"_"+"repo"
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints/fine-tune-LM",model_path)
    else:
        model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
        
    config=AutoConfig.from_pretrained(model_name)
    
    if args.model_name=="bigbird-roberta-large":
        config.block_size=16
        config.num_random_blocks=2
        config.attention_type="original_full"
        # tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=2048)
        tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
        model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings-2)
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
    test_module=utils.Loader_Creation(test_df, tokenizer,args.feature_name)

#     train_indices, val_indices=utils.mask_creation(df_train, 'label', args.seed, args.validation_split)

    

#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)

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
                                batch_size=args.test_batch_size,
                                collate_fn=train_module.collate_fn,
                                pin_memory=True,
                                # num_workers=4
                               )

    test_dataloader=DataLoader(test_module,
                                shuffle=False,
                                batch_size=args.test_batch_size,
                                collate_fn=test_module.collate_fn,
                                pin_memory=True,
                                # num_workers=4
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

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
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
    
#     model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
#         model, optimizer, train_dataloader, valid_dataloader, test_dataloader
#     )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
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

#         epoch_loss=np.mean(losses)
#         accelerator.print(f"\n** avg_loss : {epoch_loss:.2f}, time :~ {(time.time()-t0)//60} min ({time.time()-t0 :.2f} sec)***\n")


        # train_pred,train_target,train_losses=utils.eval_func(train_dataloader,
        #                                                      model, 
        #                                                      accelerator.device,
        #                                                      num_classes=num_classes, 
        #                                                      loss_weight=loss_weight)
        
        
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
        
    test_pred,test_target,test_losses=utils.eval_func(test_dataloader,
                                                      model, 
                                                      accelerator.device,
                                                      num_classes=num_classes, 
                                                      loss_weight=loss_weight)

    best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), min_recall=args.val_min_recall, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
    test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)

    if accelerator.is_main_process:
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
            
        # with open(os.path.join(output_dir,"y_true_pred.txt"),'w') as f:
        #     for x,y,z in zip(test_target.tolist(),y_pred,test_pred[:,1].tolist()):
        #         f.write(str(x)+","+str(y)+","+str(z)+ '\n')
                
        # with open("y_true_pred.txt","r") as file:
        #     true_y=[]
        #     pred_y=[]
        #     prob_y=[]
        #     for line in file:
        #         x,y,z=line.strip().split(',')
        #         true_y.append(int(x))
        #         pred_y.append(int(y))
        #         prob_y.append(float(z))
        # all(x==y for x,y in zip(true_y, test_target))
        # all(x==y for x,y in zip(pred_y, y_pred))
        # all(x==y for x,y in zip(prob_y, test_pred[:,1]))

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
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--test_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in validation")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
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
    
    args= parser.parse_args()

    if args.customized_model:
        args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_customized"
    else:
        args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]


    seed_everything(args.seed)

    print()
    print(args)
    print()
    

    data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","datasets")
    
#     train_df=load_from_disk(os.path.join(data_dir,"train_df"))
#     val_df=load_from_disk(os.path.join(data_dir,"val_df"))
#     test_df=load_from_disk(os.path.join(data_dir,"test_df"))
    
#     train_df=train_df.filter(lambda x: x['text_length']>10)
#     val_df=val_df.filter(lambda x: x['text_length']>10)
#     test_df=test_df.filter(lambda x: x['text_length']>10)
    
#     hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
#     hf_data.save_to_disk(os.path.join(data_dir,'hf_data'))
#     print(hf_data)
    # hf_data=load_from_disk(os.path.join(data_dir,"hf_data"))
    
    train_val_test=pd.read_pickle(os.path.join(data_dir,"new_train_val_test_pickle"))
    train_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="training_set"])
    val_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="validation_set"])
    test_df=datasets.Dataset.from_pandas(train_val_test[train_val_test["data_type"]=="test_set"])
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])
    
    def binary_label(example):
        return {"target": 1 if example["is_complaint"]=="Y" else 0}

    train_df=hf_data["train"].map(binary_label)
    val_df=hf_data["validation"].map(binary_label)
    test_df=hf_data["test"].map(binary_label)
    
    def under_sampling_func(df,feature,negative_positive_ratio,seed=101):
        df.set_format(type="pandas")
        data=df[:]
        df=utils.under_sampling(data,feature, seed, negative_positive_ratio)
        df.reset_index(drop=True, inplace=True)  
        df=datasets.Dataset.from_pandas(df)
        return df
    if args.train_undersampling:
        train_df=under_sampling_func(train_df,"target",args.train_negative_positive_ratio,args.seed)
    if args.val_undersampling:
        val_df=under_sampling_func(val_df,"target",args.val_negative_positive_ratio,args.seed)
    if args.test_undersampling:
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
    