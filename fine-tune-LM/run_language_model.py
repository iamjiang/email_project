import argparse
import logging
import os
import math
import copy
import numpy as np
import pandas as pd
import time
import datetime
import shutil

from dataclasses import dataclass, field
import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from datasets import ClassLabel

import transformers
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser, Trainer
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForMaskedLM
from transformers import default_data_collator
from transformers import get_scheduler
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import random
import collections
from tqdm.auto import tqdm

from IPython.display import display, HTML

print(f"Torch Version is {torch.__version__}")
print(f"Transformers Version is {transformers.__version__}")

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
    
def format_time(elapsed):
    #### Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded=int(round(elapsed)) ### round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))
 
def main(args,dataset):
    model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_name)
    config=AutoConfig.from_pretrained(model_path)
    pretained_tokenizer=AutoTokenizer.from_pretrained(model_path,model_max_length=config.max_position_embeddings-2)
    if args.customized_tokenizer:
        tokenize_dir=os.path.join(os.getcwd(),args.output_dir,"JPMC-email-tokenizer")
        tokenizer=pretained_tokenizer.from_pretrained(tokenize_dir)
    else:
        tokenizer=pretained_tokenizer
        
    model=AutoModelForMaskedLM.from_pretrained(model_path)
    print()
    print("The total # of parameters of model is {:,}".format(sum([p.nelement() for p in model.parameters()]) ) )
    print("maximal token length allowed is {:,}".format(tokenizer.model_max_length))
    print("The vocaburary size is {:,}".format(tokenizer.vocab_size))
    print()
    
    def tokenize_function(examples):
        result = tokenizer(examples["preprocessed_email"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    def group_texts(examples,chunk_size):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result
    
    ## In whold-word masking, we don't want to mask . or .. or <s> or </s>
    def whole_word_masking_data_collator(features):
        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            np.random.seed(101)
            mask = np.random.binomial(1, args.wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    ## In whold-word masking, we don't want to mask . or .. or <s> or </s>
                    ## tokenizer.convert_tokens_to_ids(["<s>","</s>",".",".."])=[0, 2, 4, 7586]
                    if input_ids[idx] not in [0,2,4,7586]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
                    
            feature['input_ids']=input_ids
            feature['labels']=new_labels

        return default_data_collator(features)
    
    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = whole_word_masking_data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["preprocessed_email"])
    tokenized_datasets=tokenized_datasets.select_columns(['input_ids', 'attention_mask', 'word_ids'])
    lm_datasets=tokenized_datasets.map(group_texts, batched=True,num_proc=args.num_workers, fn_kwargs={"chunk_size":args.chunk_size})
    
    eval_dataset = lm_datasets["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=lm_datasets["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=whole_word_masking_data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
    )
    print("")
    print(f"train_dataloader # is {len(train_dataloader)}")
    print(f"eval_dataloader # is {len(eval_dataloader)}")
    print("")
    
    optimizer=AdamW(model.parameters(),lr=5e-5)
    
    t_total = int((len(train_dataloader) // args.batch_size)//args.gradient_accumulation_steps*float(args.num_epochs))
    
    warmup_steps=int((len(train_dataloader) // args.batch_size)//args.gradient_accumulation_steps*args.warmup_ratio)
    
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

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
#     lr_scheduler =get_linear_schedule_with_warmup(optimizer, 
#                                                   num_warmup_steps=warmup_steps, 
#                                                   num_training_steps=t_total
#                                                  )
    
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=t_total)

    
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
#     model_path = f'{args.output_dir}'
#     if os.path.exists(model_path):
#         os.system(f"rm -rf {model_path}")
# #         shutil.rmtree(model_path)
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     logger.info(f'model path : {model_path}')
    
    progress_bar = tqdm(range(t_total))

    best_metric = float('inf')
    
    for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
        # Training
        
        losses=[]
        iter_tput = []
        
        model.train()
        for step,batch in enumerate(train_dataloader):
            
            tic_step = time.time()
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch={k:v.to(accelerator.device) for k,v in batch.items()}  
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
                optimizer.step()
#                 if args.use_schedule:
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            losses.append(loss.item())
            iter_tput.append(len(batch) / (time.time() - tic_step))
            
            if step%(len(train_dataloader)//10)==0 and not step==0 :
                accelerator.print('Epoch {:02d} | Step {:05d} | perplexity {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                                  .format(epoch, step, math.exp(np.mean(losses[-10:])), np.mean(iter_tput[3:]), 
                                          torch.cuda.max_memory_allocated() / 1000000))

                
        # Evaluation for Training Set 
        t1 = time.time()
        model.eval()
        train_losses = []
        for batch in tqdm(train_dataloader,position=0 ,leave=True):
            with torch.no_grad():
                outputs = model(**batch)

            train_loss = outputs.loss
            train_losses.append(accelerator.gather(train_loss.repeat(args.batch_size)))

        train_losses = torch.cat(train_losses)
        train_losses = train_losses[: len(lm_datasets["train"])]
        try:
            train_perplexity = math.exp(torch.mean(train_losses))
        except OverflowError:
            train_perplexity = float("inf")
            
        training_time=time.time()-t1
            
        # Evaluation for Validation Set 
        t2 = time.time()
        model.eval()
        val_losses = []
        for batch in tqdm(eval_dataloader,position=0 ,leave=True):
            with torch.no_grad():
                outputs = model(**batch)

            val_loss = outputs.loss
            val_losses.append(accelerator.gather(val_loss.repeat(args.batch_size)))

        val_losses = torch.cat(val_losses)
        val_losses = val_losses[: len(eval_dataset)]
        try:
            val_perplexity = math.exp(torch.mean(val_losses))
        except OverflowError:
            val_perplexity = float("inf")
            
        validation_time=time.time()-t2
        
        accelerator.print()
#         accelerator.print("Epoch {:} | Training Perplexity {:.4f} | Validation Perplexity {:.4f}"
#                           .format(epoch,train_perplexity,val_perplexity))
        accelerator.print("Epoch {:} | Training Perplexity {:.4f} | Training_Elapsed: {:} | Validation Perplexity {:.4f} | Validation_Elapsed: {:}" \
                          .format(epoch,train_perplexity,format_time(training_time),val_perplexity,format_time(validation_time)))

        accelerator.print()
        
        logger.info(accelerator.state)
        
        if args.customized_tokenizer:
            output_dir=os.path.join(os.getcwd(), args.output_dir,"JPMC-email-tokenizer")
        else:
            output_dir=os.path.join(os.getcwd(),args.output_dir)
            
        if accelerator.is_main_process:
            with open(os.path.join(output_dir,"Perplexity.txt"),'a') as f:
                f.write(f'{epoch},{train_perplexity},{val_perplexity}\n')
        
        ##### Save Checkpoint #####
        if val_perplexity<best_metric:
            logger.info(f'Performance improve after epoch: {epoch} ... ')
            best_metric=val_perplexity
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
    
if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("Roberta-Longformer Model")
    
    argparser.add_argument('--gpus', type=int, default=[0,3,5,7], nargs='+', help='used gpu')
    argparser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    argparser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")
    argparser.add_argument("--test_undersampling", action="store_true", help="undersampling or not")
    argparser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    argparser.add_argument("--val_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in validation")
    argparser.add_argument("--test_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test")
    argparser.add_argument('--chunk_size', type=int, default=512, nargs='+', help='chunk_size')
    argparser.add_argument('--num_epochs', type=int, default=12)
    argparser.add_argument("--wwm_probability", default=0.15, help="probability to mask tokens")
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument("--gradient_accumulation_steps",type=int,default=8,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    argparser.add_argument('--lr', type=float, default=3e-5)
    argparser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    argparser.add_argument('--customized_tokenizer',  action="store_true")
    argparser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    argparser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    argparser.add_argument('--use_schedule', action="store_true")  ## default type= Bool  default value=True
    argparser.add_argument("--seed",  type=int,default=101)
    argparser.add_argument('--num_workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    
    argparser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    argparser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    argparser.add_argument("--warmup_ratio", default=0.4, type=float, help="Linear warmup over warmup_steps.")
    argparser.add_argument("--num_cycles", default=2, type=int, help="Number of cycle for restart schedule with warmup.")
    
    argparser.add_argument("--model_name", default="roberta-base", type=str, help="pretrained model name")
    argparser.add_argument("--output_dir", default="roberta-base-finetuned", type=str, help="output folder name")

    
    args = argparser.parse_args()

    args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_repo"
    
    seed_everything(101)
    
#     model_name="allenai/led-large-16384"
#     model_name="allenai/led-base-16384"

    print(args)
    
    data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "email-complaints","few-shot")
    hf_data=load_from_disk(os.path.join(data_path,"hf_data"))
    hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint','target'])
    
    train_df=hf_data["train"]
    val_df=hf_data["validation"]
    test_df=hf_data["test"]

    def under_sampling(df_train,target_variable, seed, negative_positive_ratio):
        np.random.seed(seed)
        LABEL=df_train[target_variable].values.squeeze()
        IDX=np.arange(LABEL.shape[0])
        positive_idx=IDX[LABEL==1]
        negative_idx=np.random.choice(IDX[LABEL==0],size=(len(positive_idx)*negative_positive_ratio,))
        _idx=np.concatenate([positive_idx,negative_idx])
        under_sampling_train=df_train.loc[_idx,:]
        return under_sampling_train
    
    def sampling_func(df,feature,negative_positive_ratio,seed=101):
        df.set_format(type="pandas")
        data=df[:]
        df=under_sampling(data,feature, seed, negative_positive_ratio)
        df.reset_index(drop=True, inplace=True)  
        df=datasets.Dataset.from_pandas(df)
        return df

    if args.train_undersampling:
        train_df=sampling_func(hf_data["train"],"target",args.train_negative_positive_ratio,args.seed)
    if args.val_undersampling:
        val_df=sampling_func(hf_data["validation"],"target",args.val_negative_positive_ratio,args.seed)
    if args.test_undersampling:
        test_df=sampling_func(hf_data["test"],"target",args.test_negative_positive_ratio,args.seed)
    
    if args.train_undersampling or args.val_undersampling or args.test_undersampling:
        all_text=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})
    
    all_data=concatenate_datasets([all_text["train"], \
                                      all_text["validation"],\
                                      all_text["test"]
                                      ])
    
    sample_data=all_data.train_test_split(train_size=0.8,seed=101)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpus)
    print(f"The number of GPUs is {torch.cuda.device_count()}")
    
    main(args,sample_data)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()