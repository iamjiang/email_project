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

from sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    # DebertaPrefixForSequenceClassification
) 

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
    
def main(args,train_data, val_data, test_data, device):

    train_df=Dataset.from_pandas(train_data)
    val_df=Dataset.from_pandas(val_data)
    test_df=Dataset.from_pandas(test_data)
    
    model_name=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_checkpoint)
    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name,padding=True,truncation=True,model_max_length=config.max_position_embeddings-2)

    PREFIX_MODELS = {
        "bert":  BertPrefixForSequenceClassification,
        "roberta": RobertaPrefixForSequenceClassification,
        # "deberta": DebertaPrefixForSequenceClassification
    }

    PROMPT_MODELS  = {
        "bert":  BertPromptForSequenceClassification,
        "roberta": RobertaPromptForSequenceClassification
    }

    if args.prefix:
        config.hidden_dropout_prob = args.hidden_dropout_prob
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection
        config.prefix_hidden_size = args.prefix_hidden_size
        config.model_type=args.model_type
        config.device=device

        model_class=PREFIX_MODELS[config.model_type]
        model = model_class.from_pretrained(model_name,config=config)
        # model = model.to(device)

    elif args.prompt:
        config.pre_seq_len = args.pre_seq_len
        config.model_type=args.model_type
        config.device=device

        model_class=PROMPT_MODELS[config.model_type]
        model = model_class.from_pretrained(model_name,config=config)
        model = model.to(device)
    
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
                                drop_last=False   # longformer model bug
                               )
    
#     train_dataloader=DataLoader(train_module,
#                                 sampler=train_sampler,
#                                 batch_size=args.train_batch_size,
#                                 collate_fn=train_module.collate_fn,
#                                 drop_last=True   # longformer model bug
#                                )

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
    


    

    
    
    best_metric = float('inf')
    # best_metric = 0
    
    iter_tput = []
    
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
        
        print(f"\n===========EPOCH {epoch+1}/{args.num_epochs}===============\n")
        model.train()
                
        losses=[]
        for step,batch in enumerate(train_dataloader):
            t0=time.time()
            # batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            inputs={k:v[:,0:tokenizer.model_max_length-config.pre_seq_len].type(torch.LongTensor).to(device)  for k,v in batch.items() if k!="labels"}
            outputs=model(**inputs)
            logits=outputs['logits']
            # logits=outputs['logits']
            
            if loss_weight is None:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"])
            else:
                loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"], \
                                       weight=loss_weight.float().to(device)) 
            
            loss.backward()
            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
                optimizer.step()
                if args.use_schedule:
                    lr_scheduler.step()
                optimizer.zero_grad()
                
            losses.append(loss.item())
            
            iter_tput.append(batch["input_ids"].shape[0] / (time.time() - t0))
            
            if step%(len(train_dataloader)//10)==0 and not step==0 :
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                                  .format(epoch, step, np.mean(losses[-10:]), np.mean(iter_tput[3:]), 
                                          torch.cuda.max_memory_allocated() / 1000000))

# #         epoch_loss=np.mean(losses)
# #         accelerator.print(f"\n** avg_loss : {epoch_loss:.2f}, time :~ {(time.time()-t0)//60} min ({time.time()-t0 :.2f} sec)***\n")

#         t1=time.time()
#         train_pred,train_target,train_losses=utils.eval_func(train_dataloader,
#                                                              model, 
#                                                              accelerator.device,
#                                                              num_classes=num_classes, 
#                                                              loss_weight=loss_weight)

#         avg_train_loss=np.mean(train_losses)

#         train_output=utils.model_evaluate(train_target.reshape(-1),train_pred)
        
#         model_output_name=os.path.join(os.getcwd(), args.model_output_name)
#         if accelerator.is_main_process:
#             if not os.path.exists(model_output_name):
#                 os.makedirs(model_output_name)

#         t2=time.time()
#         accelerator.print("")
#         accelerator.print("==> Running Validation on training set \n")
#         accelerator.print("")
#         accelerator.print("avg_loss: {:.6f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".\
#                format(avg_train_loss, train_output["true_prediction"], train_output["false_prediction"], train_output["accuracy"], \
#                      train_output["precision"], train_output["recall"], train_output["f1_score"], train_output["AUC"], train_output["pr_auc"], \
#                       utils.format_time(t2-t1)))
#         if accelerator.is_main_process:
#             # gain_1=train_output["GAIN"]["1%"]
#             # gain_5=train_output["GAIN"]["5%"]
#             # gain_10=train_output["GAIN"]["10%"]
#             with open(os.path.join(model_output_name,"metrics_training.txt"),'a') as f:
#                 f.write(f'{args.model_checkpoint},{epoch},{avg_train_loss},{train_output["true_prediction"]},{train_output["false_prediction"]},{train_output["accuracy"]},{train_output["precision"]},{train_output["recall"]},{train_output["f1_score"]},{train_output["AUC"]},{train_output["pr_auc"]}\n')    

#         t3=time.time()
        
#         test_pred,test_target,test_losses=utils.eval_func(test_dataloader,model,accelerator.device)
#         avg_test_loss=np.mean(test_losses)
#         test_output=utils.model_evaluate(test_target.reshape(-1),test_pred)

#         t4=time.time()
#         accelerator.print("")
#         accelerator.print("==> Running Validation on test set \n")
#         accelerator.print("")
#         accelerator.print("avg_loss: {:.6f} | True_Prediction: {:,} | False_Prediction: {:,} | accuracy: {:.2%} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%} | Elapsed: {:}".\
#                format(avg_test_loss, test_output["true_prediction"], test_output["false_prediction"], test_output["accuracy"], \
#                      test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"], \
#                       utils.format_time(t4-t3)))  

#         if accelerator.is_main_process:
#             # gain_1=test_output["GAIN"]["1%"]
#             # gain_5=test_output["GAIN"]["5%"]
#             # gain_10=test_output["GAIN"]["10%"]
#             with open(os.path.join(model_output_name,"metrics_test.txt"),'a') as f:
#                 f.write(f'{args.model_checkpoint},{epoch},{avg_test_loss},{test_output["true_prediction"]},{test_output["false_prediction"]},{test_output["accuracy"]},{test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]}\n')    

#         output_dir=os.path.join(os.getcwd(), args.output_dir)
#         if accelerator.is_main_process:
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
                
#         selected_metric=avg_test_loss
#         if selected_metric<best_metric:
#             best_metric=selected_metric
#             accelerator.wait_for_everyone()
#             unwrapped_model = accelerator.unwrap_model(model)
#             unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
#             if accelerator.is_main_process:
#                 tokenizer.save_pretrained(output_dir)
#                 accelerator.print("")
#                 logger.info(f'Performance improve after epoch: {epoch+1} ... ')
#                 accelerator.print("")          
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--shuffle_train",  type=bool,default=True,help="shuffle data or not")
    parser.add_argument("--validation_split",  type=float,default=0.2,help="The split ratio for validation dataset")
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=2,help="Undersampling negative vs position ratio in training")
    parser.add_argument("--test_negative_positive_ratio",  type=int,default=10,help="Undersampling negative vs position ratio in test set")
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--truncation_strategy", type=str, default="head",help="how to truncate the long length email")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    #     parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_checkpoint', type=str, required = True)
    parser.add_argument('--model_type', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--model_output_name",  type=str, default=None)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--frozen_layers", type=int, default=0,help="freeze layers without gradient updates")
    parser.add_argument("--frozen_all", action='store_true', default=0,help="freeze all parameters")
    
    parser.add_argument('--prefix', action="store_true",help="Will use P-tuning v2 during training")
    parser.add_argument('--prefix_model', type=str, default="bert", help="the back-bone model for prefix tuning")
    parser.add_argument('--prompt', action="store_true",help="Will use prompt tuning during training")
    parser.add_argument('--pre_seq_len', type=int, default=4,help="The length of prompt")
    parser.add_argument('--prefix_projection', action="store_true",help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument('--prefix_hidden_size', type=int, default=512,help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="The dropout probability used in the models")
    
    args= parser.parse_args()

    # args.model_output_name=f'{args.model_output_name}_{args.truncation_strategy}'
    # args.output_dir=f'{args.output_dir}_{args.truncation_strategy}'
    args.output_dir=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1] + "_repo"
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
    if args.undersampling:
        train_df=under_sampling_func(train_df,"target",args.train_negative_positive_ratio,args.seed)
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
    