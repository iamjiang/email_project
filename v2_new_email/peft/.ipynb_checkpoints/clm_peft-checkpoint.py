# import sys
# sys.path.append('/opt/omniai/work/instance1/jupyter/new-email-project/peft/src')
# sys.path.append('/opt/omniai/work/instance1/jupyter/new-email-project/peft')
# sys.path=list(set(sys.path))

# path_to_remove=["/opt/omniai/work/instance1/jupyter/email-complaints/peft/","/opt/omniai/work/instance1/jupyter/email-complaints/peft/src/"]
# for path in path_to_remove:
#     if path in sys.path:
#         sys.path.remove(path)

import argparse
import os
import time
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.mapping import get_peft_config, get_peft_model
from src.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from src.config import PeftType, TaskType, PeftConfig, PromptLearningConfig
from src.peft_model import PeftModel
from src.prefix_tuning import PrefixTuningConfig
from src.prompt_tuning import PromptTuningConfig
from src.p_tuning import PromptEncoderConfig

import evaluate
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    AdamW,
    get_linear_schedule_with_warmup,
    get_scheduler,
    set_seed
)

from accelerate import Accelerator

from tqdm import tqdm

import bitsandbytes as bnb

import utils

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main(args, train_data, val_data, device):
    
    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)

    config=AutoConfig.from_pretrained(model_path)

    peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, encoder_hidden_size=128)
    
    if any(k in args.model_name.split("-") for k in ["gpt", "opt", "bloom","GPT"]):
        padding_side = "left"
    else:
        padding_side = "right"

    if any(k in args.model_name.split("-") for k in ["bloom"]):
        tokenizer = AutoTokenizer.from_pretrained(model_path, \
                                              padding_side=padding_side,\
                                              model_max_length=args.max_length-args.num_virtual_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, \
                                                  padding_side=padding_side,\
                                                  model_max_length=config.max_position_embeddings-2-args.num_virtual_tokens)
    
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True)
    model = get_peft_model(model, peft_config)

    if tokenizer.model_max_length:
        model_max_length=tokenizer.model_max_length
    else:
        model_max_length=args.max_length
        
    print()
    print(f"The maximal # input tokens : {model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    All_params=sum([p.nelement() for p in model.parameters()])
    Trainable_params=sum([p.nelement() for p in model.parameters() if p.requires_grad==True])
    print(f"Trainable params : {Trainable_params:,}")
    print(f"All params : {All_params:,}")
    print(f"Trainable percentage : {Trainable_params/All_params:.4%}")
    print()
    print(model.peft_config)
    print()
    
    # text_label=train_data['text_label'].unique().tolist()
    train_module=utils.Loader_Creation(hf_data["train"], tokenizer,"preprocessed_email",model_max_length, args.text_label)
    val_module=utils.Loader_Creation(hf_data["validation"], tokenizer,"preprocessed_email",model_max_length, args.text_label)
    # test_module=utils.Loader_Creation(hf_data["test"], tokenizer,"preprocessed_email",model_max_length,text_label)

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
                                collate_fn=val_module.collate_fn,
                                pin_memory=True,
                                # num_workers=4
                               )
    
#     test_dataloader=DataLoader(test_module,
#                                shuffle=False,
#                                batch_size=args.val_batch_size,
#                                collate_fn=train_module.collate_fn,
#                                pin_memory=True,
#                                # num_workers=4
#                                )

    print()
    print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
    print('{:<30}{:<10,} '.format("validation mini-batch",len(valid_dataloader)))
    # print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))
    print()

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
        config.use_cache=False
        

    model, optimizer, train_dataloader,valid_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader )

    best_metric = float('inf')
    # best_metric = 0
    best_epoch = 0

    iter_tput = []
    
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):

        accelerator.print(f"\n===========EPOCH {epoch+1}/{args.num_epochs}===============\n")
        
        model.train()
        total_loss=0
        losses=[]
        for step,batch in enumerate(train_dataloader):
            t0=time.time()
            # keep_mask = batch["labels"].ne(-100).any(dim=0)
            # mask_pos=torch.where(keep_mask)[0].tolist()
            batch={k:v.type(torch.LongTensor).to(accelerator.device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss+=loss.detach().float()
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
                    

        model.eval()
        eval_preds=[]
        eval_loss = 0
        for step, batch in enumerate(tqdm(valid_dataloader,position=0 ,leave=True)):
            batch={k:v.type(torch.LongTensor).to(accelerator.device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(valid_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        
        if args.deduped:
            output_dir=os.path.join(os.getcwd(), args.output_dir,"dedup")
        else:
            output_dir=os.path.join(os.getcwd(), args.output_dir)
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(os.path.join(output_dir,"train_val_ppl.txt"),'a') as f:
            f.write(f'{args.model_name},{train_epoch_loss},{train_ppl},{eval_epoch_loss}, {eval_ppl}\n')   
            
        peft_model_id = f"{args.model_name}_{peft_config.peft_type}_{peft_config.task_type}"
        
        selected_metric=eval_ppl
        if selected_metric<best_metric:
            best_metric=selected_metric
            best_epoch = epoch
            model.save_pretrained(os.path.join(output_dir,peft_model_id))
            
        #Early stopping
        if epoch - best_epoch > args.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, best_metric))
            break 

    
    config = PeftConfig.from_pretrained(output_dir,peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(output_dir,model, peft_model_id)

#     model.to(device)
#     model.eval()
    
#     fin_targets=[]
#     fin_outputs=[]
#     for step, batch in enumerate(tqdm(test_dataloader,position=0 ,leave=True)):
#         X=[]
#         with torch.no_grad():
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             outputs = model.generate(input_ids=inputs["input_ids"], \
#                                      attention_mask=inputs["attention_mask"], \
#                                      max_new_tokens=2, \
#                                      eos_token_id=tokenizer.pad_token_id
#                                     )
#             text_ouput=tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
            
#             for result in text_output: 
#                 if result[0].split(":")[-1].strip()=="complaint":
#                 x.extend([1])
#             else:
#                 x.extend([0]) 

#         fin_targets.append(batch["labels"].cpu().detach().numpy())
#         fin_outputs.append
        
#     best_threshold=utils.find_optimal_threshold(val_target.squeeze(), val_pred[:,1].squeeze(), \
#                                                    min_recall=args.val_min_recall, pos_label=False)
#     y_pred=[1 if x>best_threshold else 0 for x in val_pred[:,1]]
#     val_output=utils.model_evaluate(val_target.reshape(-1),y_pred)

#     if accelerator.is_main_process:
#         with open(os.path.join(output_dir,"metrics_val.txt"),'a') as f:
#             f.write(f'{args.model_name},{val_output["total positive"]},{val_output["false positive"]},\
#             {val_output["false_negative"]}, {val_output["precision"]},{val_output["recall"]},{val_output["f1_score"]},\
#             {val_output["AUC"]},{val_output["pr_auc"]},{best_threshold}\n')       

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--gpus', type=int, default=[0,1], nargs='+', help='used gpu')
    parser.add_argument("--loss_weight", action='store_true', help="weight for unbalance data")
    parser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    parser.add_argument("--train_negative_positive_ratio",  type=int,default=5)
    parser.add_argument("--val_negative_positive_ratio",  type=int,default=5)

    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                               help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gradient_checkpointing',  action="store_true")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    # parser.add_argument('--customized_model',  action="store_true")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")    
    parser.add_argument('--use_schedule', action="store_true")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument('--model_name', type=str, default = "roberta-large")
    parser.add_argument("--output_dir", type=str, default=None, help="output folder name")
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    parser.add_argument("--frozen_layers", type=int, default=0,help="freeze layers without gradient updates")
    parser.add_argument("--frozen_all", action='store_true', default=0,help="freeze all parameters")
    parser.add_argument("--hidden_dropout_prob", default=0.2, type=float, help="dropout rate for hidden state.")
    parser.add_argument("--es_patience", type=int, default=3,
                            help="Early stopping's parameter: number of epochs with no \
                            improvement after which training will be stopped. Set to 0 to disable this technique.")

    parser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--peft_type", default="P-tuning", type=str, help="prompt tuning type")
    parser.add_argument("--num_virtual_tokens", default=20, type=int)
    parser.add_argument('--deduped', action="store_true", help="keep most recent thread id or not")
    parser.add_argument('--text_label', nargs='+', default=[" no complaint"," complaint"])
    
    parser.add_argument("--max_length", type=int, default=4096, help="set maximal text input length if tokenizer.model_max_length is not available")
    
    args= parser.parse_args()
    
    args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]
    
    print()
    print(args)
    print()

    seed_everything(args.seed)
    
    if args.peft_type=="P-tuning":
        peft_type = PeftType.P_TUNING
    elif args.peft_type=="Prefix-tuning":
        peft_type = PeftType.PREFIX_TUNING
    elif args.peft_type=="Prompt_Tuning":
        peft_type = PeftType.PROMPT_TUNING
    else:
        raise ValueError("peft_type has to be P-tuning, prefix-tuning or prompt tuning.")

    if args.deduped:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_dedup_data")
    else:
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "new-email-project","datasets","split_data")
        
    data_name=[x for x in os.listdir(data_path) if x.split("_")[-2]=="pickle"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_path,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[-1],x.shape[0]))
    
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace = True) 
    ## train: 09/2022 ~ 01/2023. validation: 02/2023  test: 03/2023
    set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1]) \
    else ("val" if (row["year"]==2023 and row["month"]==2) else "test")
    df["data_type"]=df.progress_apply(set_categories,axis=1)

    df.loc[:,'target']=df.loc[:,'is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    train_df=datasets.Dataset.from_pandas(df[df["data_type"]=="train"])
    val_df=datasets.Dataset.from_pandas(df[df["data_type"]=="val"])
    test_df=datasets.Dataset.from_pandas(df[df["data_type"]=="test"])
    hf_data=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    hf_data=hf_data.select_columns(['snapshot_id','thread_id','time','is_feedback',\
                                    'preprocessed_email','is_complaint','target'])

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
    if args.val_undersampling:
        val_df=under_sampling_func(val_df,"target",args.val_negative_positive_ratio,args.seed)
        
    hf_data=DatasetDict({"train":train_df, "validation":val_df})
    
    
    # classes=[" no complaint"," complaint"]
    classes=args.text_label
    hf_data = hf_data.map(
        lambda x: {"text_label": [classes[label] for label in x["target"]]},batched=True,num_proc=1)
    
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
    
    # test_data=hf_data['test'].shuffle(seed=101).select(range(len(hf_data["test"])))
    # test_data.set_format(type="pandas")
    # test_data=test_data[:]
    # # val_data=val_data.sample(500)    
    
    main(args,train_data, val_data,  device)