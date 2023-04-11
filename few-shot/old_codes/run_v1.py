import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import datasets
from datasets import load_dataset, load_from_disk,load_metric, concatenate_datasets, DatasetDict, Dataset
from datasets import disable_caching, enable_caching
enable_caching()
from transformers import AutoModelForMaskedLM , AutoTokenizer, AutoConfig
from transformers import AdamW, get_scheduler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wrapper
from arg_parser import arg_parse
import model_prompt 
import utils

args=arg_parse()
args.train_batch_size=4
args.test_batch_size=16
args.num_epochs=3
args.loss_weight=True
args.use_schedule=True
args.train_undersampling=True
args.val_undersampling=True
args.test_undersampling=True
args.model_checkpoint="longformer-large-4096"
args.gradient_accumulation_steps=4
args.train_negative_positive_ratio=5
args.val_negative_positive_ratio=5
args.test_negative_positive_ratio=15
args.output_dir=args.model_checkpoint.split("-")[0] + "_" + args.model_checkpoint.split("-")[1] + "_repo"
args.lr=5e-5
print(args)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models",args.model_checkpoint)
config=AutoConfig.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_length=config.max_position_embeddings-2)
print(tokenizer.mask_token)        

prompting= model_prompt.Prompting(model=model_path)

prefix_prompt="email review:"
post_prompt=f".This email is  {tokenizer.mask_token}"
prompt=[prefix_prompt,post_prompt]
token_list1=["positive"]
token_list2=["negative"]
benchmark=prompting.benchmark_prompt(prompt, token_list1, token_list2, device)
print("{:<30}{:}".format("benchmark probability",benchmark))

# hf_data=load_from_disk(os.path.join(os.getcwd(),"hf_data"))
# hf_data=hf_data.select_columns(['snapshot_id', 'preprocessed_email','is_complaint'])

# def binary_label(example):
#     return {"target": 1 if example["is_complaint"]=="Y" else 0}

# train_df=hf_data["train"].map(binary_label)
# val_df=hf_data["validation"].map(binary_label)
# test_df=hf_data["test"].map(binary_label)

# wrapped_train=wrapper.prompt_wrapper(train_df, prompt, tokenizer)
# wrapped_val=wrapper.prompt_wrapper(val_df, prompt, tokenizer)
# wrapped_test=wrapper.prompt_wrapper(test_df, prompt, tokenizer)

# wrapped_text=DatasetDict({"train":wrapped_train, "validation":wrapped_val, "test":wrapped_test})

# wrapped_text.save_to_disk(os.path.join(os.getcwd(),'wrapped_text'))
# print(wrapped_text)

wrapped_text=load_from_disk(os.path.join(os.getcwd(),"wrapped_text"))

train_df=wrapped_text["train"]
val_df=wrapped_text["validation"]
test_df=wrapped_text["test"]

def under_sampling_func(df,feature,negative_positive_ratio,seed=101):
    df.set_format(type="pandas")
    data=df[:]
    df=utils.under_sampling(data,feature, seed, negative_positive_ratio)
    df.reset_index(drop=True, inplace=True)  
    df=datasets.Dataset.from_pandas(df)
    return df
if args.train_undersampling:
    train_df=under_sampling_func(wrapped_text["train"],"target",args.train_negative_positive_ratio,args.seed)
if args.val_undersampling:
    val_df=under_sampling_func(wrapped_text["validation"],"target",args.val_negative_positive_ratio,args.seed)
if args.test_undersampling:
    test_df=under_sampling_func(wrapped_text["test"],"target",args.test_negative_positive_ratio,args.seed)

if args.train_undersampling or args.val_undersampling or args.test_undersampling:
    wrapped_text=DatasetDict({"train":train_df, "validation":val_df, "test":test_df})

print()
print(wrapped_text)
print()

train_module=wrapper.Loader_Creation(wrapped_text["train"], prompting.tokenizer,'wrapped_email')
val_module=wrapper.Loader_Creation(wrapped_text["validation"], prompting.tokenizer,'wrapped_email')
test_module=wrapper.Loader_Creation(wrapped_text["test"], prompting.tokenizer,'wrapped_email')

train_dataloader=DataLoader(train_module,
                            shuffle=False,
                            batch_size=args.train_batch_size,
                            collate_fn=train_module.collate_fn,
                            drop_last=False   # longformer model bug
                           )

val_dataloader=DataLoader(val_module,
                            shuffle=False,
                            batch_size=args.train_batch_size,
                            collate_fn=train_module.collate_fn,
                            drop_last=False   # longformer model bug
                           )

test_dataloader=DataLoader(test_module,
                            shuffle=False,
                            batch_size=args.test_batch_size,
                            collate_fn=train_module.collate_fn,
                            drop_last=False   # longformer model bug
                           )

print()
print('{:<30}{:<10,} '.format("training mini-batch",len(train_dataloader)))
print('{:<30}{:<10,} '.format("validation mini-batch",len(val_dataloader)))
print('{:<30}{:<10,} '.format("test mini-batch",len(test_dataloader)))

train_label=np.array(wrapped_text["train"]['target']).squeeze()
num_classes=np.unique(train_label).shape[0]
if args.loss_weight:
    train_classes_num, train_classes_weight = utils.get_class_count_and_weight(train_label,num_classes)
    loss_weight=torch.tensor(train_classes_weight).to(device)
    class_weights=class_weight.compute_class_weight(class_weight="balanced",classes=np.unique(train_label),y=wrapped_text["train"]['target'])
    class_weights=torch.tensor(class_weights).to(device)
    assert torch.all(torch.eq(loss_weight,class_weights))
    
else:
    loss_weight=None
    class_weights=None

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
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, 
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=t_total)


goodToken="positive"
badToken="negative"
good=prompting.tokenizer.convert_tokens_to_ids(goodToken)
bad=prompting.tokenizer.convert_tokens_to_ids(badToken)

best_metric = float('inf')
# best_metric = 0
    
iter_tput = []
for epoch in tqdm(range(args.num_epochs),position=0 ,leave=True):
    model.train()
    model=model.to(device)
    losses=[]
    for step , batch in enumerate(train_dataloader):
        t0=time.time()
        mask_pos=[v.tolist().index(tokenizer.mask_token_id) for v in batch["input_ids"]]
        mask_pos=torch.tensor(mask_pos).to(device)
        batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
        outputs=model(batch['input_ids'])
        predictions = outputs[0]
        pred=predictions.gather(1, mask_pos.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,prompting.tokenizer.vocab_size)).squeeze(1) ## dim=batch_size * vocab_size
        logits=pred[:,[good,bad]] ## dim=batch_size * 2
        prob=torch.nn.functional.softmax(logits, dim=1)
        # logits=prob/(benchmark.expand_as(prob).to(device))
        if loss_weight is None:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"])
        else:
            loss = F.cross_entropy(logits.view(-1, num_classes).to(device),batch["labels"], \
                                   weight=loss_weight.float().to(device)) 

        loss.backward()
        optimizer.step()
            
        if (step+1)%args.gradient_accumulation_steps == 0 or step==len(train_dataloader)-1:
            optimizer.step()
            if args.use_schedule:
                lr_scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        iter_tput.append(batch["input_ids"].shape[0] / (time.time() - t0)) 
        
        if step%(len(train_dataloader)//10)==0 and not step==0 :
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.2f} | GPU{:.0f} MB'
                      .format(epoch, step, np.mean(losses[-10:]), np.mean(iter_tput[3:]), torch.cuda.max_memory_allocated() / 1000000))
                
                
                
                
    # train_pred,train_target,train_losses=utils.eval_func(train_dataloader,
    #                                                      model, 
    #                                                      tokenizer,
    #                                                      device,
    #                                                      num_classes=num_classes, 
    #                                                      loss_weight=loss_weight,
    #                                                      goodToken="positive",
    #                                                      badToken="negative")




    val_pred,val_target,val_losses=utils.eval_func(val_dataloader,
                                                   model, 
                                                   tokenizer,
                                                   device,
                                                   num_classes=num_classes, 
                                                   loss_weight=loss_weight,
                                                   goodToken="positive",
                                                   badToken="negative")

    avg_val_loss=np.mean(val_losses)
    
    output_dir=os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)        
    avg_val_loss=np.mean(val_losses)
    selected_metric=avg_val_loss
    if selected_metric<best_metric:
        best_metric=selected_metric
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
            

config=AutoConfig.from_pretrained(output_dir)
model = AutoModelForMaskedLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir,model_max_length=config.max_position_embeddings-2)



test_pred,test_target,test_losses=utils.eval_func(test_dataloader,
                                                     model, 
                                                     tokenizer,
                                                     device,
                                                     num_classes=num_classes, 
                                                     loss_weight=loss_weight,
                                                     goodToken="positive",
                                                     badToken="negative")

# best_threshold=prompting.benchmark_prompt(prompt, token_list1, token_list2, device)[1].item()

best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=0.80, pos_label=False)
y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
test_output=utils.model_evaluate(test_target.reshape(-1),y_pred)

print("==> performance on test set \n")
print("")
print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".\
       format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
             test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))


print()
print(f"\n===========Test Set Performance===============\n")
print()
y_pred=[1 if x>best_threshold else 0 for x in test_pred[:,1]]
print(classification_report(test_target, y_pred))
print()
print(confusion_matrix(test_target, y_pred))  
